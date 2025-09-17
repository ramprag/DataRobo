import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import re
import hashlib
from faker import Faker
import random

logger = logging.getLogger(__name__)

class PrivacyMasker:
    """Apply privacy masking and anonymization techniques to sensitive data"""

    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)
        random.seed(42)

        # PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }

    def apply_privacy_masks(self, df: pd.DataFrame, privacy_config) -> pd.DataFrame:
        """Apply privacy masking based on configuration"""

        masked_df = df.copy()

        logger.info("Applying privacy masks to dataset")

        # Apply masking based on configuration
        if privacy_config.mask_emails:
            masked_df = self._mask_emails(masked_df)

        if privacy_config.mask_names:
            masked_df = self._mask_names(masked_df)

        if privacy_config.mask_phone_numbers:
            masked_df = self._mask_phone_numbers(masked_df)

        if privacy_config.mask_addresses:
            masked_df = self._mask_addresses(masked_df)

        if privacy_config.mask_ssn:
            masked_df = self._mask_ssn(masked_df)

        # Apply masking to custom fields
        if privacy_config.custom_fields:
            for field in privacy_config.custom_fields:
                if field in masked_df.columns:
                    masked_df = self._mask_column(
                        masked_df,
                        field,
                        privacy_config.anonymization_method
                    )

        logger.info("Privacy masking completed")
        return masked_df

    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the type of PII in a column"""

        # Get sample of non-null string values
        sample_values = series.dropna().astype(str).head(20).tolist()

        if not sample_values:
            return 'unknown'

        # Count pattern matches
        pattern_matches = {}
        for pii_type, pattern in self.pii_patterns.items():
            match_count = sum(1 for val in sample_values if re.search(pattern, val))
            if match_count > 0:
                pattern_matches[pii_type] = match_count / len(sample_values)

        # Return the pattern with highest match rate
        if pattern_matches:
            return max(pattern_matches, key=pattern_matches.get)

        return 'unknown'

    def _mask_emails(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask email addresses"""

        email_columns = self._find_columns_by_pattern(df, 'email')

        for column in email_columns:
            df[column] = df[column].apply(self._mask_email_value)

        if email_columns:
            logger.info(f"Masked email columns: {email_columns}")

        return df

    def _mask_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask name fields"""

        name_patterns = ['name', 'first_name', 'last_name', 'full_name', 'firstname', 'lastname']
        name_columns = []

        for column in df.columns:
            column_lower = column.lower().replace(' ', '_')
            if any(pattern in column_lower for pattern in name_patterns):
                name_columns.append(column)

        for column in name_columns:
            if 'first' in column.lower():
                df[column] = df[column].apply(lambda x: self.fake.first_name() if pd.notna(x) else x)
            elif 'last' in column.lower():
                df[column] = df[column].apply(lambda x: self.fake.last_name() if pd.notna(x) else x)
            else:
                df[column] = df[column].apply(lambda x: self.fake.name() if pd.notna(x) else x)

        if name_columns:
            logger.info(f"Masked name columns: {name_columns}")

        return df

    def _mask_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask phone numbers"""

        phone_columns = self._find_columns_by_pattern(df, 'phone')

        for column in phone_columns:
            df[column] = df[column].apply(self._mask_phone_value)

        if phone_columns:
            logger.info(f"Masked phone columns: {phone_columns}")

        return df

    def _mask_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask address fields"""

        address_patterns = ['address', 'street', 'city', 'state', 'zip', 'postal', 'country']
        address_columns = []

        for column in df.columns:
            column_lower = column.lower().replace(' ', '_')
            if any(pattern in column_lower for pattern in address_patterns):
                address_columns.append(column)

        for column in address_columns:
            column_lower = column.lower()
            if 'street' in column_lower or 'address' in column_lower:
                df[column] = df[column].apply(lambda x: self.fake.street_address() if pd.notna(x) else x)
            elif 'city' in column_lower:
                df[column] = df[column].apply(lambda x: self.fake.city() if pd.notna(x) else x)
            elif 'state' in column_lower:
                df[column] = df[column].apply(lambda x: self.fake.state() if pd.notna(x) else x)
            elif 'zip' in column_lower or 'postal' in column_lower:
                df[column] = df[column].apply(lambda x: self.fake.zipcode() if pd.notna(x) else x)
            elif 'country' in column_lower:
                df[column] = df[column].apply(lambda x: self.fake.country() if pd.notna(x) else x)

        if address_columns:
            logger.info(f"Masked address columns: {address_columns}")

        return df

    def _mask_ssn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask Social Security Numbers"""

        ssn_columns = self._find_columns_by_pattern(df, 'ssn')

        for column in ssn_columns:
            df[column] = df[column].apply(self._mask_ssn_value)

        if ssn_columns:
            logger.info(f"Masked SSN columns: {ssn_columns}")

        return df

    def _find_columns_by_pattern(self, df: pd.DataFrame, pattern_type: str) -> List[str]:
        """Find columns that match PII patterns"""

        matching_columns = []

        # First, check column names
        name_patterns = {
            'email': ['email', 'e_mail', 'mail', 'email_address'],
            'phone': ['phone', 'telephone', 'mobile', 'cell', 'phone_number'],
            'ssn': ['ssn', 'social_security', 'social_security_number', 'tax_id']
        }

        if pattern_type in name_patterns:
            for column in df.columns:
                column_lower = column.lower().replace(' ', '_')
                if any(pattern in column_lower for pattern in name_patterns[pattern_type]):
                    matching_columns.append(column)

        # Then, check data patterns for remaining columns
        if pattern_type in self.pii_patterns:
            pattern = self.pii_patterns[pattern_type]

            for column in df.columns:
                if column not in matching_columns:
                    # Sample some values to check for pattern
                    sample_values = df[column].dropna().astype(str).head(10).tolist()
                    match_count = sum(1 for val in sample_values if re.search(pattern, val))

                    # If more than 50% of sample matches, consider it a match
                    if sample_values and match_count / len(sample_values) > 0.5:
                        matching_columns.append(column)

        return matching_columns

    def _mask_column(self, df: pd.DataFrame, column: str, method: str = 'faker') -> pd.DataFrame:
        """Apply masking to a specific column"""

        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataset")
            return df

        if method == 'hash':
            df[column] = df[column].apply(self._hash_value)
        elif method == 'redact':
            df[column] = df[column].apply(self._redact_value)
        else:  # faker method
            df[column] = df[column].apply(lambda x: self._generate_faker_replacement(x, column))

        return df

    def _mask_email_value(self, value):
        """Mask a single email value"""
        if pd.isna(value):
            return value

        value_str = str(value)
        if re.match(self.pii_patterns['email'], value_str):
            return self.fake.email()

        return value

    def _mask_phone_value(self, value):
        """Mask a single phone value"""
        if pd.isna(value):
            return value

        value_str = str(value)
        if re.search(self.pii_patterns['phone'], value_str):
            return self.fake.phone_number()

        return value

    def _mask_ssn_value(self, value):
        """Mask a single SSN value"""
        if pd.isna(value):
            return value

        value_str = str(value)
        if re.match(self.pii_patterns['ssn'], value_str):
            return self.fake.ssn()

        return value

    def _hash_value(self, value):
        """Hash a value using SHA256"""
        if pd.isna(value):
            return value

        value_str = str(value)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]  # Truncate for readability

    def _redact_value(self, value):
        """Redact a value with asterisks"""
        if pd.isna(value):
            return value

        value_str = str(value)
        if len(value_str) <= 2:
            return '*' * len(value_str)
        else:
            return value_str[0] + '*' * (len(value_str) - 2) + value_str[-1]

    def _generate_faker_replacement(self, value, column_name: str):
        """Generate a Faker replacement based on column context"""
        if pd.isna(value):
            return value

        column_lower = column_name.lower()

        # Try to match appropriate faker method based on column name
        if 'email' in column_lower:
            return self.fake.email()
        elif 'name' in column_lower:
            return self.fake.name()
        elif 'phone' in column_lower:
            return self.fake.phone_number()
        elif 'address' in column_lower:
            return self.fake.address().replace('\n', ', ')
        elif 'city' in column_lower:
            return self.fake.city()
        elif 'company' in column_lower:
            return self.fake.company()
        else:
            # For generic fields, try to preserve data type
            if isinstance(value, (int, float)):
                return self.fake.random_number(digits=len(str(int(value))))
            else:
                return self.fake.word()

    def generate_privacy_report(self, original_df: pd.DataFrame, masked_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a report on privacy masking applied"""

        report = {
            'total_columns': len(original_df.columns),
            'columns_masked': 0,
            'masking_details': {},
            'pii_detected': {},
            'masking_timestamp': pd.Timestamp.now().isoformat()
        }

        # Compare original and masked data
        for column in original_df.columns:
            original_values = set(original_df[column].dropna().astype(str))
            masked_values = set(masked_df[column].dropna().astype(str))

            # Check if values changed (indicating masking)
            if original_values != masked_values:
                report['columns_masked'] += 1

                # Detect PII type
                pii_type = self._detect_column_type(original_df[column])

                report['masking_details'][column] = {
                    'pii_type': pii_type,
                    'original_unique_values': len(original_values),
                    'masked_unique_values': len(masked_values),
                    'values_changed': len(original_values - masked_values),
                    'masking_rate': len(original_values - masked_values) / len(original_values) if original_values else 0
                }

                if pii_type != 'unknown':
                    if pii_type not in report['pii_detected']:
                        report['pii_detected'][pii_type] = []
                    report['pii_detected'][pii_type].append(column)

        report['privacy_score'] = self._calculate_privacy_score(report)

        return report

    def _calculate_privacy_score(self, report: Dict[str, Any]) -> float:
        """Calculate a privacy protection score (0-100)"""

        if report['total_columns'] == 0:
            return 100.0

        # Base score
        masking_ratio = report['columns_masked'] / report['total_columns']
        base_score = masking_ratio * 60  # Up to 60 points for masking coverage

        # Bonus for PII detection and masking
        pii_bonus = min(len(report['pii_detected']) * 10, 40)  # Up to 40 points for PII handling

        # Quality bonus based on masking effectiveness
        quality_bonus = 0
        for details in report['masking_details'].values():
            if details['masking_rate'] > 0.9:  # High masking rate
                quality_bonus += 2

        privacy_score = min(base_score + pii_bonus + quality_bonus, 100.0)

        return round(privacy_score, 1)
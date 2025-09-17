import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""

    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with error handling"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode file with any supported encoding")

            # Basic data validation
            if df.empty:
                raise ValueError("Uploaded file is empty")

            # Log basic statistics
            logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

            return df

        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze data types and provide metadata about each column"""
        analysis = {}

        for column in df.columns:
            col_data = df[column]
            analysis[column] = {
                'dtype': str(col_data.dtype),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': float(col_data.isnull().sum() / len(col_data) * 100),
                'unique_values': int(col_data.nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(col_data),
                'is_categorical': pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() < 20,
                'sample_values': col_data.dropna().head(5).tolist() if not col_data.empty else []
            }

            # Additional analysis for numeric columns
            if analysis[column]['is_numeric']:
                analysis[column].update({
                    'min': float(col_data.min()) if not col_data.isnull().all() else None,
                    'max': float(col_data.max()) if not col_data.isnull().all() else None,
                    'mean': float(col_data.mean()) if not col_data.isnull().all() else None,
                    'std': float(col_data.std()) if not col_data.isnull().all() else None
                })

        return analysis

    def detect_pii_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect potential PII columns based on column names and data patterns"""
        pii_patterns = {
            'email': ['email', 'e_mail', 'mail', 'email_address'],
            'name': ['name', 'first_name', 'last_name', 'full_name', 'firstname', 'lastname'],
            'phone': ['phone', 'telephone', 'mobile', 'cell', 'phone_number'],
            'address': ['address', 'street', 'city', 'zip', 'postal', 'country', 'state'],
            'ssn': ['ssn', 'social_security', 'social_security_number', 'tax_id'],
            'id': ['id', 'user_id', 'customer_id', 'account_id', 'identifier'],
            'date_of_birth': ['dob', 'birth_date', 'date_of_birth', 'birthday'],
            'credit_card': ['card', 'credit_card', 'card_number', 'cc_number']
        }

        detected_pii = {category: [] for category in pii_patterns.keys()}

        for column in df.columns:
            column_lower = column.lower().replace(' ', '_')

            for category, patterns in pii_patterns.items():
                if any(pattern in column_lower for pattern in patterns):
                    detected_pii[category].append(column)
                    break

            # Additional pattern matching for data content
            if not any(column in cols for cols in detected_pii.values()):
                sample_data = df[column].dropna().astype(str).head(10).tolist()

                # Email pattern detection
                if any('@' in str(val) and '.' in str(val) for val in sample_data):
                    detected_pii['email'].append(column)

                # Phone number pattern detection
                elif any(len(str(val).replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) >= 10
                         and str(val).replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit()
                         for val in sample_data):
                    detected_pii['phone'].append(column)

        # Remove empty categories
        detected_pii = {k: v for k, v in detected_pii.items() if v}

        logger.info(f"Detected PII columns: {detected_pii}")
        return detected_pii

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        df_clean = df.copy()

        # Remove completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')

        # Convert data types appropriately
        for column in df_clean.columns:
            # Try to convert to numeric if mostly numeric
            if df_clean[column].dtype == 'object':
                numeric_count = pd.to_numeric(df_clean[column], errors='coerce').notna().sum()
                total_count = df_clean[column].notna().sum()

                if total_count > 0 and numeric_count / total_count > 0.8:
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')

        logger.info("Data cleaning completed")
        return df_clean

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return metrics"""
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'completely_null_columns': int((df.isnull().all()).sum()),
            'data_types': df.dtypes.value_counts().to_dict(),
            'quality_score': 0.0
        }

        # Calculate quality score (0-100)
        quality_score = 100.0

        # Penalize for missing data
        if metrics['missing_data_percentage'] > 50:
            quality_score -= 30
        elif metrics['missing_data_percentage'] > 20:
            quality_score -= 15
        elif metrics['missing_data_percentage'] > 5:
            quality_score -= 5

        # Penalize for duplicate rows
        duplicate_percentage = metrics['duplicate_rows'] / metrics['total_rows'] * 100
        if duplicate_percentage > 20:
            quality_score -= 20
        elif duplicate_percentage > 5:
            quality_score -= 10

        # Penalize for completely null columns
        if metrics['completely_null_columns'] > 0:
            quality_score -= metrics['completely_null_columns'] * 5

        metrics['quality_score'] = max(0, quality_score)

        return metrics

    def prepare_for_synthesis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for synthetic generation"""
        df_prepared = df.copy()

        # Handle missing values
        for column in df_prepared.columns:
            if df_prepared[column].dtype in ['object', 'string']:
                # Fill categorical/string columns with most frequent value
                mode_value = df_prepared[column].mode()
                if not mode_value.empty:
                    df_prepared[column] = df_prepared[column].fillna(mode_value.iloc[0])
                else:
                    df_prepared[column] = df_prepared[column].fillna('Unknown')
            else:
                # Fill numeric columns with median
                median_value = df_prepared[column].median()
                if not pd.isna(median_value):
                    df_prepared[column] = df_prepared[column].fillna(median_value)
                else:
                    df_prepared[column] = df_prepared[column].fillna(0)

        logger.info("Data prepared for synthesis")
        return df_prepared

    def export_data(self, df: pd.DataFrame, file_path: str, format: str = 'csv') -> bool:
        """Export data to specified format"""
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            elif format.lower() == 'xlsx':
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Data exported to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
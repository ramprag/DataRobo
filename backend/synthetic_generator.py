import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

# SDV imports
try:
    from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN
    from sdv.constraints import GreaterThan, Between, Positive, Negative
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    logging.warning("SDV not available. Using fallback synthetic generation.")

# Faker for realistic fake data
from faker import Faker
import random

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic data while preserving statistical properties and relationships"""

    def __init__(self, method: str = "auto"):
        self.method = method
        self.fake = Faker()
        self.model = None
        self.constraints = []

        # Set random seeds for reproducibility
        Faker.seed(42)
        random.seed(42)
        np.random.seed(42)

    def generate_synthetic_data(
            self,
            df: pd.DataFrame,
            num_rows: int = None,
            method: str = None
    ) -> pd.DataFrame:
        """Generate synthetic data using the best available method"""

        if num_rows is None:
            num_rows = len(df)

        method = method or self.method

        logger.info(f"Generating {num_rows} synthetic rows using method: {method}")

        # Auto-select method based on data characteristics
        if method == "auto":
            method = self._select_best_method(df)

        try:
            if method == "sdv_gaussian" and SDV_AVAILABLE:
                return self._generate_with_sdv_gaussian(df, num_rows)
            elif method == "sdv_ctgan" and SDV_AVAILABLE:
                return self._generate_with_sdv_ctgan(df, num_rows)
            elif method == "statistical":
                return self._generate_statistical(df, num_rows)
            else:
                return self._generate_faker_based(df, num_rows)

        except Exception as e:
            logger.error(f"Error with method {method}: {e}")
            logger.info("Falling back to faker-based generation")
            return self._generate_faker_based(df, num_rows)

    def _select_best_method(self, df: pd.DataFrame) -> str:
        """Select the best synthetic generation method based on data characteristics"""

        if not SDV_AVAILABLE:
            return "statistical"

        # Factors to consider
        num_rows = len(df)
        num_cols = len(df.columns)
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = num_cols - numeric_cols

        # For small datasets, use statistical method
        if num_rows < 100:
            return "statistical"

        # For datasets with many categorical variables, use CTGAN
        if categorical_cols / num_cols > 0.7:
            return "sdv_ctgan"

        # For mostly numeric data, use Gaussian Copula
        if numeric_cols / num_cols > 0.7:
            return "sdv_gaussian"

        # Default to Gaussian Copula for balanced datasets
        return "sdv_gaussian"

    def _generate_with_sdv_gaussian(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using SDV Gaussian Copula"""
        logger.info("Using SDV Gaussian Copula method")

        # Prepare constraints
        constraints = self._create_constraints(df)

        # Create and train model
        model = GaussianCopula(constraints=constraints)
        model.fit(df)

        # Generate synthetic data
        synthetic_df = model.sample(num_rows)

        # Post-process to ensure data quality
        synthetic_df = self._post_process_synthetic_data(df, synthetic_df)

        return synthetic_df

    def _generate_with_sdv_ctgan(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using SDV CTGAN"""
        logger.info("Using SDV CTGAN method")

        # Create and train model
        model = CTGAN(epochs=100)  # Reduced epochs for faster processing
        model.fit(df)

        # Generate synthetic data
        synthetic_df = model.sample(num_rows)

        # Post-process to ensure data quality
        synthetic_df = self._post_process_synthetic_data(df, synthetic_df)

        return synthetic_df

    def _generate_statistical(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using statistical distributions"""
        logger.info("Using statistical distribution method")

        synthetic_data = {}

        for column in df.columns:
            col_data = df[column].dropna()

            if col_data.empty:
                synthetic_data[column] = [None] * num_rows
                continue

            if pd.api.types.is_numeric_dtype(col_data):
                # Generate numeric data using normal distribution
                mean = col_data.mean()
                std = col_data.std()

                if std == 0 or pd.isna(std):
                    # If no variation, use constant value
                    synthetic_data[column] = [mean] * num_rows
                else:
                    # Generate from normal distribution with clipping to original range
                    min_val, max_val = col_data.min(), col_data.max()
                    synthetic_values = np.random.normal(mean, std, num_rows)
                    synthetic_values = np.clip(synthetic_values, min_val, max_val)

                    # Preserve integer type if original was integer
                    if col_data.dtype in ['int32', 'int64']:
                        synthetic_values = synthetic_values.astype(int)

                    synthetic_data[column] = synthetic_values.tolist()

            elif pd.api.types.is_datetime64_any_dtype(col_data):
                # Generate datetime data
                min_date = col_data.min()
                max_date = col_data.max()
                date_range = (max_date - min_date).days

                if date_range == 0:
                    synthetic_data[column] = [min_date] * num_rows
                else:
                    random_days = np.random.randint(0, date_range, num_rows)
                    synthetic_dates = [min_date + timedelta(days=int(days)) for days in random_days]
                    synthetic_data[column] = synthetic_dates

            else:
                # Generate categorical data using frequency distribution
                value_counts = col_data.value_counts(normalize=True)
                values = value_counts.index.tolist()
                probabilities = value_counts.values

                synthetic_data[column] = np.random.choice(
                    values, size=num_rows, p=probabilities
                ).tolist()

        return pd.DataFrame(synthetic_data)

    def _generate_faker_based(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using Faker for realistic fake data"""
        logger.info("Using Faker-based method")

        synthetic_data = {}

        for column in df.columns:
            col_data = df[column].dropna()

            if col_data.empty:
                synthetic_data[column] = [None] * num_rows
                continue

            column_lower = column.lower()

            # Generate based on column name patterns
            if any(pattern in column_lower for pattern in ['name', 'first_name', 'lastname']):
                synthetic_data[column] = [self.fake.name() for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['email', 'mail']):
                synthetic_data[column] = [self.fake.email() for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['phone', 'telephone']):
                synthetic_data[column] = [self.fake.phone_number() for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['address', 'street']):
                synthetic_data[column] = [self.fake.address().replace('\n', ', ') for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['city']):
                synthetic_data[column] = [self.fake.city() for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['country']):
                synthetic_data[column] = [self.fake.country() for _ in range(num_rows)]
            elif any(pattern in column_lower for pattern in ['company', 'employer']):
                synthetic_data[column] = [self.fake.company() for _ in range(num_rows)]
            elif pd.api.types.is_numeric_dtype(col_data):
                # Numeric data with distribution matching
                min_val, max_val = col_data.min(), col_data.max()
                if col_data.dtype in ['int32', 'int64']:
                    synthetic_data[column] = [self.fake.random_int(min=int(min_val), max=int(max_val)) for _ in range(num_rows)]
                else:
                    synthetic_data[column] = [self.fake.random.uniform(min_val, max_val) for _ in range(num_rows)]
            else:
                # Categorical data using frequency distribution
                value_counts = col_data.value_counts(normalize=True)
                values = value_counts.index.tolist()
                probabilities = value_counts.values

                synthetic_data[column] = np.random.choice(
                    values, size=num_rows, p=probabilities
                ).tolist()

        return pd.DataFrame(synthetic_data)

    def _create_constraints(self, df: pd.DataFrame) -> List:
        """Create constraints for SDV models"""
        constraints = []

        for column in df.select_dtypes(include=[np.number]).columns:
            col_data = df[column].dropna()
            if not col_data.empty:
                min_val = col_data.min()
                max_val = col_data.max()

                if min_val > 0:
                    constraints.append(Positive(column_names=[column]))
                elif max_val < 0:
                    constraints.append(Negative(column_names=[column]))
                else:
                    constraints.append(Between(
                        column_names=[column],
                        low=min_val,
                        high=max_val
                    ))

        return constraints

    def _post_process_synthetic_data(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process synthetic data to ensure quality and consistency"""

        processed_df = synthetic_df.copy()

        # Ensure data types match original
        for column in processed_df.columns:
            if column in original_df.columns:
                original_dtype = original_df[column].dtype

                try:
                    if original_dtype in ['int32', 'int64']:
                        processed_df[column] = processed_df[column].round().astype(original_dtype)
                    elif original_dtype == 'bool':
                        processed_df[column] = processed_df[column].astype(bool)
                    elif pd.api.types.is_categorical_dtype(original_dtype):
                        # Ensure categorical values are within original categories
                        original_categories = original_df[column].cat.categories
                        processed_df[column] = pd.Categorical(
                            processed_df[column],
                            categories=original_categories
                        )
                except Exception as e:
                    logger.warning(f"Could not convert {column} to original dtype: {e}")

        # Clip numeric values to original ranges
        for column in processed_df.select_dtypes(include=[np.number]).columns:
            if column in original_df.columns:
                original_col = original_df[column].dropna()
                if not original_col.empty:
                    min_val, max_val = original_col.min(), original_col.max()
                    processed_df[column] = processed_df[column].clip(min_val, max_val)

        # Handle any remaining NaN values
        for column in processed_df.columns:
            if processed_df[column].isnull().any():
                if column in original_df.columns:
                    # Fill with most common value from original data
                    most_common = original_df[column].mode()
                    if not most_common.empty:
                        processed_df[column] = processed_df[column].fillna(most_common.iloc[0])

        return processed_df

    def get_generation_metadata(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Get metadata about the synthetic data generation process"""

        metadata = {
            'generation_timestamp': datetime.utcnow().isoformat(),
            'method_used': self.method,
            'original_shape': original_df.shape,
            'synthetic_shape': synthetic_df.shape,
            'columns_generated': list(synthetic_df.columns),
            'data_types_preserved': {},
            'value_ranges_preserved': {}
        }

        # Check data type preservation
        for column in synthetic_df.columns:
            if column in original_df.columns:
                orig_dtype = str(original_df[column].dtype)
                synth_dtype = str(synthetic_df[column].dtype)
                metadata['data_types_preserved'][column] = {
                    'original': orig_dtype,
                    'synthetic': synth_dtype,
                    'preserved': orig_dtype == synth_dtype
                }

        # Check value range preservation for numeric columns
        for column in synthetic_df.select_dtypes(include=[np.number]).columns:
            if column in original_df.columns:
                orig_min, orig_max = original_df[column].min(), original_df[column].max()
                synth_min, synth_max = synthetic_df[column].min(), synthetic_df[column].max()

                metadata['value_ranges_preserved'][column] = {
                    'original_range': [float(orig_min), float(orig_max)],
                    'synthetic_range': [float(synth_min), float(synth_max)],
                    'range_preserved': synth_min >= orig_min and synth_max <= orig_max
                }

        return metadata
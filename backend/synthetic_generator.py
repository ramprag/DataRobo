import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from faker import Faker
import random

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic data while preserving statistical properties"""

    def __init__(self, method: str = "statistical"):
        self.method = method
        self.fake = Faker()

        # Set random seeds for reproducibility
        Faker.seed(42)
        random.seed(42)
        np.random.seed(42)

    def generate_synthetic_data(
            self,
            df: pd.DataFrame,
            num_rows: int = None
    ) -> pd.DataFrame:
        """Generate synthetic data using statistical distributions"""
        try:
            if num_rows is None:
                num_rows = len(df)

            logger.info(f"Generating {num_rows} synthetic rows using statistical method")

            synthetic_data = {}

            for column in df.columns:
                try:
                    col_data = df[column].dropna()

                    if col_data.empty:
                        synthetic_data[column] = [None] * num_rows
                        continue

                    if pd.api.types.is_numeric_dtype(col_data):
                        # Generate numeric data using normal distribution
                        synthetic_data[column] = self._generate_numeric_column(col_data, num_rows)
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        # Generate datetime data
                        synthetic_data[column] = self._generate_datetime_column(col_data, num_rows)
                    else:
                        # Generate categorical data using frequency distribution
                        synthetic_data[column] = self._generate_categorical_column(col_data, num_rows)

                except Exception as e:
                    logger.warning(f"Error generating synthetic data for column {column}: {e}")
                    # Fallback: repeat original values
                    if len(df) > 0:
                        synthetic_data[column] = np.random.choice(df[column].fillna('Unknown'), size=num_rows).tolist()
                    else:
                        synthetic_data[column] = ['Unknown'] * num_rows

            result_df = pd.DataFrame(synthetic_data)
            logger.info(f"Successfully generated synthetic dataset with {len(result_df)} rows and {len(result_df.columns)} columns")
            return result_df

        except Exception as e:
            logger.error(f"Error in synthetic data generation: {e}")
            raise

    def _generate_numeric_column(self, col_data: pd.Series, num_rows: int) -> List:
        """Generate numeric synthetic data"""
        try:
            mean = col_data.mean()
            std = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()

            if std == 0 or pd.isna(std):
                # If no variation, use constant value with small random noise
                values = [mean + np.random.normal(0, 0.01) for _ in range(num_rows)]
            else:
                # Generate from normal distribution with clipping to original range
                values = np.random.normal(mean, std, num_rows)
                values = np.clip(values, min_val, max_val)

            # Preserve integer type if original was integer
            if col_data.dtype in ['int32', 'int64', 'Int32', 'Int64']:
                values = [int(round(v)) for v in values]

            return values.tolist() if hasattr(values, 'tolist') else values

        except Exception as e:
            logger.warning(f"Error in numeric generation: {e}")
            # Fallback: sample from original values
            return np.random.choice(col_data.values, size=num_rows, replace=True).tolist()

    def _generate_categorical_column(self, col_data: pd.Series, num_rows: int) -> List:
        """Generate categorical synthetic data"""
        try:
            # Use frequency distribution
            value_counts = col_data.value_counts(normalize=True)
            values = value_counts.index.tolist()
            probabilities = value_counts.values

            if len(values) == 0:
                return ['Unknown'] * num_rows

            synthetic_values = np.random.choice(
                values, size=num_rows, p=probabilities, replace=True
            ).tolist()

            return synthetic_values

        except Exception as e:
            logger.warning(f"Error in categorical generation: {e}")
            # Fallback: sample uniformly from unique values
            unique_values = col_data.unique()
            if len(unique_values) > 0:
                return np.random.choice(unique_values, size=num_rows, replace=True).tolist()
            else:
                return ['Unknown'] * num_rows

    def _generate_datetime_column(self, col_data: pd.Series, num_rows: int) -> List:
        """Generate datetime synthetic data"""
        try:
            min_date = col_data.min()
            max_date = col_data.max()
            date_range = (max_date - min_date).days

            if date_range <= 0:
                return [min_date] * num_rows

            # Generate random dates within the original range
            random_days = np.random.randint(0, date_range + 1, num_rows)
            synthetic_dates = [min_date + timedelta(days=int(days)) for days in random_days]

            return synthetic_dates

        except Exception as e:
            logger.warning(f"Error in datetime generation: {e}")
            # Fallback: sample from original values
            return np.random.choice(col_data.dropna().values, size=num_rows, replace=True).tolist()

    def get_generation_metadata(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Get metadata about the synthetic data generation process"""
        try:
            metadata = {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'method_used': self.method,
                'original_shape': original_df.shape,
                'synthetic_shape': synthetic_df.shape,
                'columns_generated': list(synthetic_df.columns),
                'generation_successful': True
            }

            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'method_used': self.method,
                'generation_successful': False,
                'error': str(e)
            }
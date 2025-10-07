import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from faker import Faker
import random
import os
import pickle

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic data with GAN model support and statistical fallback"""

    def __init__(self, method: str = "auto"):
        """
        Initialize synthetic data generator

        Args:
            method: "auto" (default), "gan", "ctgan", or "statistical"
        """
        self.method = method
        self.fake = Faker()
        self.gan_available = False
        self.ctgan_model = None
        self.tvae_model = None

        # Set random seeds for reproducibility
        Faker.seed(42)
        random.seed(42)
        np.random.seed(42)

        # Try to import GAN libraries
        self._initialize_gan_support()

    def _initialize_gan_support(self):
        """Initialize GAN model support if available"""
        try:
            from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
            from sdv.metadata import SingleTableMetadata

            self.CTGANSynthesizer = CTGANSynthesizer
            self.TVAESynthesizer = TVAESynthesizer
            self.SingleTableMetadata = SingleTableMetadata
            self.gan_available = True

            logger.info("✅ GAN support initialized successfully (SDV library available)")

        except ImportError as e:
            logger.warning(f"⚠️ GAN libraries not available: {e}")
            logger.warning("Falling back to statistical method. Install with: pip install sdv")
            self.gan_available = False

    def generate_synthetic_data(
            self,
            df: pd.DataFrame,
            num_rows: int = None,
            use_gan: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic data using GAN (if available) or statistical method

        Args:
            df: Original dataframe
            num_rows: Number of rows to generate (None = same as original)
            use_gan: Whether to attempt GAN generation (default: True)

        Returns:
            Synthetic dataframe
        """
        try:
            if num_rows is None:
                num_rows = len(df)

            # Decide which method to use
            effective_method = self._determine_method(df, use_gan)

            logger.info(f"🎯 Using {effective_method} method to generate {num_rows} rows")

            if effective_method in ["ctgan", "tvae"]:
                return self._generate_with_gan(df, num_rows, effective_method)
            else:
                return self._generate_statistical(df, num_rows)

        except Exception as e:
            logger.error(f"❌ Error in synthetic data generation: {e}")
            logger.warning("⚠️ Falling back to statistical method")
            return self._generate_statistical(df, num_rows)

    def _determine_method(self, df: pd.DataFrame, use_gan: bool) -> str:
        """Determine which generation method to use"""

        # If user explicitly set method, respect it
        if self.method != "auto":
            if self.method in ["gan", "ctgan", "tvae"] and not self.gan_available:
                logger.warning(f"⚠️ {self.method} requested but not available, using statistical")
                return "statistical"
            return self.method if self.method != "gan" else "ctgan"

        # If GAN not requested, use statistical
        if not use_gan:
            return "statistical"

        # If GAN not available, use statistical
        if not self.gan_available:
            return "statistical"

        # Check if dataset is suitable for GAN
        if len(df) < 100:
            logger.info("📊 Dataset too small for GAN (<100 rows), using statistical method")
            return "statistical"

        if len(df.columns) > 100:
            logger.info("📊 Too many columns for GAN (>100), using statistical method")
            return "statistical"

        # Default to CTGAN for suitable datasets
        logger.info("✨ Dataset suitable for GAN, using CTGAN")
        return "ctgan"

    def _generate_with_gan(
            self,
            df: pd.DataFrame,
            num_rows: int,
            model_type: str = "ctgan"
    ) -> pd.DataFrame:
        """
        Generate synthetic data using GAN models (CTGAN or TVAE)

        Args:
            df: Original dataframe
            num_rows: Number of rows to generate
            model_type: "ctgan" or "tvae"
        """
        try:
            logger.info(f"🤖 Starting {model_type.upper()} training...")

            # Prepare data
            df_clean = self._prepare_data_for_gan(df)

            # Create metadata
            metadata = self.SingleTableMetadata()
            metadata.detect_from_dataframe(df_clean)

            # Log detected column types
            logger.info(f"📋 Detected columns: {list(df_clean.columns)}")

            # Initialize model
            if model_type == "ctgan":
                synthesizer = self.CTGANSynthesizer(
                    metadata,
                    epochs=100,  # Reduced for faster training
                    verbose=True,
                    cuda=False  # Set to True if GPU available
                )
            else:  # tvae
                synthesizer = self.TVAESynthesizer(
                    metadata,
                    epochs=100,
                    verbose=True,
                    cuda=False
                )

            # Train the model
            logger.info(f"🎓 Training {model_type.upper()} model...")
            synthesizer.fit(df_clean)

            # Generate synthetic data
            logger.info(f"✨ Generating {num_rows} synthetic rows...")
            synthetic_df = synthesizer.sample(num_rows=num_rows)

            # Restore original column order and types
            synthetic_df = self._post_process_gan_data(synthetic_df, df)

            logger.info(f"✅ {model_type.upper()} generation completed successfully!")
            return synthetic_df

        except Exception as e:
            logger.error(f"❌ {model_type.upper()} generation failed: {e}")
            logger.warning("⚠️ Falling back to statistical method")
            raise

    def _prepare_data_for_gan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for GAN training"""
        df_clean = df.copy()

        # Handle missing values - fill with mode/median
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)

        # Convert datetime columns to numerical representation
        for col in df_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype('int64') // 10**9  # Convert to unix timestamp

        return df_clean

    def _post_process_gan_data(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process GAN-generated data to match original format"""

        # Ensure column order matches original
        synthetic_df = synthetic_df[original_df.columns]

        # Restore data types
        for col in original_df.columns:
            if col in synthetic_df.columns:
                # Restore integer types
                if original_df[col].dtype in ['int32', 'int64', 'Int32', 'Int64']:
                    synthetic_df[col] = synthetic_df[col].round().astype('int64')

                # Restore datetime types
                elif pd.api.types.is_datetime64_any_dtype(original_df[col]):
                    synthetic_df[col] = pd.to_datetime(synthetic_df[col], unit='s', errors='coerce')

        return synthetic_df

    def _generate_statistical(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using statistical distributions (fallback method)"""
        try:
            logger.info(f"📊 Generating {num_rows} synthetic rows using statistical method")

            synthetic_data = {}

            for column in df.columns:
                try:
                    col_data = df[column].dropna()

                    if col_data.empty:
                        synthetic_data[column] = [None] * num_rows
                        continue

                    if pd.api.types.is_numeric_dtype(col_data):
                        synthetic_data[column] = self._generate_numeric_column(col_data, num_rows)
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        synthetic_data[column] = self._generate_datetime_column(col_data, num_rows)
                    else:
                        synthetic_data[column] = self._generate_categorical_column(col_data, num_rows)

                except Exception as e:
                    logger.warning(f"⚠️ Error generating synthetic data for column {column}: {e}")
                    if len(df) > 0:
                        synthetic_data[column] = np.random.choice(df[column].fillna('Unknown'), size=num_rows).tolist()
                    else:
                        synthetic_data[column] = ['Unknown'] * num_rows

            result_df = pd.DataFrame(synthetic_data)
            logger.info(f"✅ Statistical generation completed: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df

        except Exception as e:
            logger.error(f"❌ Error in statistical generation: {e}")
            raise

    def _generate_numeric_column(self, col_data: pd.Series, num_rows: int) -> List:
        """Generate numeric synthetic data"""
        try:
            mean = col_data.mean()
            std = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()

            if std == 0 or pd.isna(std):
                values = [mean + np.random.normal(0, 0.01) for _ in range(num_rows)]
            else:
                values = np.random.normal(mean, std, num_rows)
                values = np.clip(values, min_val, max_val)

            if col_data.dtype in ['int32', 'int64', 'Int32', 'Int64']:
                values = [int(round(v)) for v in values]

            return values.tolist() if hasattr(values, 'tolist') else values

        except Exception as e:
            logger.warning(f"⚠️ Error in numeric generation: {e}")
            return np.random.choice(col_data.values, size=num_rows, replace=True).tolist()

    def _generate_categorical_column(self, col_data: pd.Series, num_rows: int) -> List:
        """Generate categorical synthetic data"""
        try:
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
            logger.warning(f"⚠️ Error in categorical generation: {e}")
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

            random_days = np.random.randint(0, date_range + 1, num_rows)
            synthetic_dates = [min_date + timedelta(days=int(days)) for days in random_days]

            return synthetic_dates

        except Exception as e:
            logger.warning(f"⚠️ Error in datetime generation: {e}")
            return np.random.choice(col_data.dropna().values, size=num_rows, replace=True).tolist()

    def get_generation_metadata(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Get metadata about the synthetic data generation process"""
        try:
            metadata = {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'method_used': self.method if self.method != "auto" else ("gan" if self.gan_available else "statistical"),
                'gan_available': self.gan_available,
                'original_shape': original_df.shape,
                'synthetic_shape': synthetic_df.shape,
                'columns_generated': list(synthetic_df.columns),
                'generation_successful': True
            }

            return metadata

        except Exception as e:
            logger.error(f"❌ Error generating metadata: {e}")
            return {
                'generation_timestamp': datetime.utcnow().isoformat(),
                'method_used': self.method,
                'generation_successful': False,
                'error': str(e)
            }

    def get_available_methods(self) -> List[str]:
        """Get list of available generation methods"""
        methods = ["statistical"]
        if self.gan_available:
            methods.extend(["ctgan", "tvae", "gan"])
        return methods
"""
GAN-Based Synthetic Data Generator using CTGAN
Replaces the statistical approach with true GAN implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GANSyntheticDataGenerator:
    """
    GAN-based synthetic data generator using CTGAN (Conditional Tabular GAN)
    Supports both single-table and multi-table generation with relationship preservation
    """

    def __init__(self, use_ctgan: bool = True, epochs: int = 300, batch_size: int = 500):
        """
        Initialize GAN generator

        Args:
            use_ctgan: If True, use CTGAN. If False, fallback to statistical method
            epochs: Number of training epochs for GAN
            batch_size: Batch size for training
        """
        self.use_ctgan = use_ctgan
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = {}  # Store trained models per table

        # Try to import CTGAN
        try:
            from ctgan import CTGAN
            from ctgan import load_demo
            self.CTGAN = CTGAN
            self.ctgan_available = True
            logger.info("✓ CTGAN library loaded successfully")
        except ImportError:
            logger.warning("⚠ CTGAN not installed. Install with: pip install ctgan")
            self.ctgan_available = False
            self.use_ctgan = False

    def generate_synthetic_data(self, df: pd.DataFrame, num_rows: Optional[int] = None,
                                preserve_columns: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic data using GAN or fallback method

        Args:
            df: Original dataframe
            num_rows: Number of rows to generate (None = same as original)
            preserve_columns: Columns to preserve (PK/FK that should not be generated)

        Returns:
            Synthetic dataframe
        """
        if num_rows is None:
            num_rows = len(df)

        preserve_columns = preserve_columns or []

        # Separate preserved columns from generated columns
        preserve_df = df[preserve_columns].copy() if preserve_columns else None
        generate_cols = [col for col in df.columns if col not in preserve_columns]

        if not generate_cols:
            # All columns preserved, just return original
            return df.copy()

        generate_df = df[generate_cols].copy()

        # Use GAN if available and enabled
        if self.use_ctgan and self.ctgan_available:
            try:
                synthetic_df = self._generate_with_ctgan(generate_df, num_rows)
            except Exception as e:
                import traceback
                logger.error(f"CTGAN generation failed: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.warning("Falling back to statistical method")
                synthetic_df = self._generate_statistical(generate_df, num_rows)
        else:
            synthetic_df = self._generate_statistical(generate_df, num_rows)

        # Combine preserved and generated columns
        if preserve_df is not None and len(preserve_df) > 0:
            # If we need more rows, cycle through preserved data
            if num_rows > len(preserve_df):
                preserve_df = pd.concat([preserve_df] * (num_rows // len(preserve_df) + 1),
                                       ignore_index=True).head(num_rows)
            elif num_rows < len(preserve_df):
                preserve_df = preserve_df.head(num_rows)

            result_df = pd.concat([preserve_df.reset_index(drop=True),
                                  synthetic_df.reset_index(drop=True)], axis=1)
        else:
            result_df = synthetic_df

        # Ensure column order matches original
        result_df = result_df[df.columns]

        return result_df

    def _generate_with_ctgan(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using CTGAN"""
        logger.info(f"🎲 Generating {num_rows} rows using CTGAN (GAN-based approach)")
        logger.info(f"   Training on {len(df)} original rows with {len(df.columns)} columns")

        # Prepare data - handle data types
        df_prepared = self._prepare_data_for_ctgan(df)

        # Detect discrete columns (categorical)
        discrete_columns = self._detect_discrete_columns(df_prepared)
        logger.info(f"   Detected {len(discrete_columns)} discrete columns: {discrete_columns}")

        # Initialize and train CTGAN
        logger.info(f"   Training CTGAN model (epochs={self.epochs}, batch_size={self.batch_size})...")
        start_time = datetime.now()

        ctgan = self.CTGAN(
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(df)),
            verbose=False
        )

        ctgan.fit(df_prepared, discrete_columns=discrete_columns)

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"   ✓ CTGAN training completed in {training_time:.2f}s")

        # Generate synthetic data
        logger.info(f"   Generating {num_rows} synthetic samples...")
        synthetic_df = ctgan.sample(num_rows)

        # Post-process to ensure data quality
        synthetic_df = self._post_process_ctgan_output(synthetic_df, df)

        logger.info(f"   ✓ CTGAN generation completed")

        return synthetic_df

    def _prepare_data_for_ctgan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for CTGAN training"""
        df_prepared = df.copy()

        for col in df_prepared.columns:
            # Handle datetime
            if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
                df_prepared[col] = df_prepared[col].astype(str)

            # Handle boolean
            elif pd.api.types.is_bool_dtype(df_prepared[col]):
                df_prepared[col] = df_prepared[col].astype(int)

            # Fill NaN values
            if df_prepared[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_prepared[col]):
                    df_prepared[col].fillna(df_prepared[col].median(), inplace=True)
                else:
                    df_prepared[col].fillna('Unknown', inplace=True)

        return df_prepared

    def _detect_discrete_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect discrete (categorical) columns for CTGAN"""
        discrete_columns = []

        for col in df.columns:
            # If string/object type, it's discrete
            if df[col].dtype == 'object':
                discrete_columns.append(col)
            # If numeric but few unique values, treat as discrete
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05:  # Less than 5% unique values
                    discrete_columns.append(col)

        return discrete_columns

    def _post_process_ctgan_output(self, synthetic_df: pd.DataFrame,
                                   original_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process CTGAN output to ensure data quality"""
        for col in synthetic_df.columns:
            # Clip numeric values to original range
            if pd.api.types.is_numeric_dtype(original_df[col]):
                min_val = original_df[col].min()
                max_val = original_df[col].max()
                synthetic_df[col] = synthetic_df[col].clip(min_val, max_val)

                # Round integers
                if pd.api.types.is_integer_dtype(original_df[col]):
                    synthetic_df[col] = synthetic_df[col].round().astype(original_df[col].dtype)

            # Ensure categorical values are from original set
            elif synthetic_df[col].dtype == 'object':
                original_categories = set(original_df[col].dropna().unique())

                # Map any generated values to nearest original category
                def map_to_original(val):
                    if pd.isna(val) or val in original_categories:
                        return val
                    # If not in original, randomly pick one
                    return np.random.choice(list(original_categories))

                synthetic_df[col] = synthetic_df[col].apply(map_to_original)

        return synthetic_df

    def _generate_statistical(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Fallback statistical method (non-GAN)"""
        logger.info(f"📊 Generating {num_rows} rows using Statistical method (fallback)")

        synthetic_data = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                synthetic_data[column] = self._generate_numeric_column(df[column], num_rows)
            else:
                synthetic_data[column] = self._generate_categorical_column(df[column], num_rows)

        return pd.DataFrame(synthetic_data)

    def _generate_numeric_column(self, series: pd.Series, num_rows: int) -> np.ndarray:
        """Generate numeric column using statistical properties"""
        clean_data = series.dropna()

        if len(clean_data) == 0:
            return np.zeros(num_rows)

        if pd.api.types.is_integer_dtype(series):
            # For integers, use discrete distribution
            mean = clean_data.mean()
            std = clean_data.std()
            synthetic = np.random.normal(mean, std, num_rows)
            synthetic = np.round(synthetic).astype(int)

            # Clip to original range
            synthetic = np.clip(synthetic, clean_data.min(), clean_data.max())
        else:
            # For floats, use continuous distribution
            mean = clean_data.mean()
            std = clean_data.std()
            synthetic = np.random.normal(mean, std, num_rows)
            synthetic = np.clip(synthetic, clean_data.min(), clean_data.max())

        return synthetic

    def _generate_categorical_column(self, series: pd.Series, num_rows: int) -> np.ndarray:
        """Generate categorical column preserving distribution"""
        clean_data = series.dropna()

        if len(clean_data) == 0:
            return np.array(['Unknown'] * num_rows)

        # Get value counts and probabilities
        value_counts = clean_data.value_counts()
        categories = value_counts.index.tolist()
        probabilities = (value_counts / len(clean_data)).tolist()

        # Sample according to original distribution
        synthetic = np.random.choice(categories, size=num_rows, p=probabilities)

        return synthetic

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the generator"""
        return {
            'type': 'CTGAN' if (self.use_ctgan and self.ctgan_available) else 'Statistical',
            'ctgan_available': self.ctgan_available,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'models_trained': len(self.models)
        }
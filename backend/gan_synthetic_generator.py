# backend/gan_synthetic_generator.py
"""
GAN-Based Synthetic Data Generator using CTGAN
- Always attempt GAN when selected and available.
- Hard per-table timeout with automatic fallback to statistical if CTGAN runs too long.
- Light adaptiveness for speed, while honoring user choice to use GAN.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logger = logging.getLogger(__name__)

class GANSyntheticDataGenerator:
    """
    GAN-based synthetic data generator using CTGAN (Conditional Tabular GAN)
    """

    def __init__(self, use_ctgan: bool = True, epochs: int = 300, batch_size: int = 256,
                 max_train_seconds: int = 120):
        """
        Args:
            use_ctgan: If True, try CTGAN; else use statistical.
            epochs: Upper bound on epochs (adaptive logic may lower this).
            batch_size: Upper bound on batch size (adaptive logic may lower this).
            max_train_seconds: Hard time budget per table for CTGAN training; if exceeded -> fallback.
        """
        self.use_ctgan = use_ctgan
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_train_seconds = max_train_seconds
        self.models = {}

        try:
            from ctgan import CTGAN
            self.CTGAN = CTGAN
            self.ctgan_available = True
            logger.info("✓ CTGAN library loaded successfully")
        except ImportError:
            logger.warning("⚠ CTGAN not installed. Install with: pip install ctgan")
            self.ctgan_available = False
            self.use_ctgan = False

        np.random.seed(42)
        random.seed(42)

    def generate_synthetic_data(self, df: pd.DataFrame, num_rows: Optional[int] = None,
                                preserve_columns: List[str] = None) -> pd.DataFrame:
        if num_rows is None:
            num_rows = len(df)

        preserve_columns = preserve_columns or []
        preserve_df = df[preserve_columns].copy() if preserve_columns else None
        generate_cols = [col for col in df.columns if col not in preserve_columns]
        if not generate_cols:
            return df.copy()

        generate_df = df[generate_cols].copy()

        if self.use_ctgan and self.ctgan_available:
            try:
                synthetic_df = self._generate_with_ctgan_timeout(generate_df, num_rows)
            except Exception as e:
                import traceback
                logger.error(f"CTGAN generation failed: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("Falling back to statistical method")
                synthetic_df = self._generate_statistical(generate_df, num_rows)
        else:
            synthetic_df = self._generate_statistical(generate_df, num_rows)

        if preserve_df is not None and len(preserve_df) > 0:
            if num_rows > len(preserve_df):
                preserve_df = pd.concat([preserve_df] * (num_rows // len(preserve_df) + 1),
                                        ignore_index=True).head(num_rows)
            else:
                preserve_df = preserve_df.head(num_rows)
            result_df = pd.concat([preserve_df.reset_index(drop=True),
                                   synthetic_df.reset_index(drop=True)], axis=1)
        else:
            result_df = synthetic_df

        return result_df[df.columns]

    def _adaptive_epochs(self, df: pd.DataFrame) -> int:
        rows, cols = len(df), len(df.columns)
        if rows <= 1000:
            base = 30
        elif rows <= 5000:
            base = 50
        else:
            base = 80
        base += min(20, max(0, (cols - 10)))
        return int(min(self.epochs, base))

    def _adaptive_batch_size(self, df: pd.DataFrame) -> int:
        rows = len(df)
        return int(min(self.batch_size, max(64, min(512, rows // 4 if rows > 0 else 256))))

    def _cap_training_rows(self, df_prepared: pd.DataFrame) -> pd.DataFrame:
        max_train_rows = 2000
        if len(df_prepared) > max_train_rows:
            return df_prepared.sample(n=max_train_rows, random_state=42)
        return df_prepared

    def _reduce_categorical_cardinality(self, df: pd.DataFrame, max_uniques: int = 100) -> pd.DataFrame:
        df2 = df.copy()
        for col in df2.columns:
            if df2[col].dtype == 'object':
                vc = df2[col].value_counts()
                if len(vc) > max_uniques:
                    keep = set(vc.head(max_uniques).index)
                    df2[col] = df2[col].apply(lambda x: x if x in keep else 'Other')
        return df2

    def _generate_with_ctgan_timeout(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """
        Train CTGAN with a hard timeout. If training exceeds max_train_seconds, raise to fallback.
        """
        logger.info(f"🎲 CTGAN target rows={num_rows}, input rows={len(df)}, cols={len(df.columns)}")

        df_prepared = self._prepare_data_for_ctgan(df)
        df_prepared = self._reduce_categorical_cardinality(df_prepared, max_uniques=100)
        df_train = self._cap_training_rows(df_prepared)

        discrete_columns = self._detect_discrete_columns(df_train)
        epochs = self._adaptive_epochs(df_train)
        batch_size = self._adaptive_batch_size(df_train)
        logger.info(f"   Plan: epochs={epochs}, batch_size={batch_size}, train_rows={len(df_train)}, timeout={self.max_train_seconds}s")

        def train_and_sample():
            ctgan = self.CTGAN(
                epochs=epochs,
                batch_size=min(batch_size, max(1, len(df_train))),
                verbose=False
            )
            ctgan.fit(df_train, discrete_columns=discrete_columns)
            out = ctgan.sample(num_rows)
            return self._post_process_ctgan_output(out, df)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_and_sample)
            try:
                start = datetime.now()
                synthetic_df = future.result(timeout=self.max_train_seconds)
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f"   ✓ CTGAN finished in {elapsed:.1f}s")
                return synthetic_df
            except FuturesTimeout:
                future.cancel()
                logger.warning(f"⏱ CTGAN training exceeded {self.max_train_seconds}s, falling back")
                raise TimeoutError("CTGAN training timed out")

    def _prepare_data_for_ctgan(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prepared = df.copy()
        for col in df_prepared.columns:
            if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
                df_prepared[col] = df_prepared[col].astype(str)
            elif pd.api.types.is_bool_dtype(df_prepared[col]):
                df_prepared[col] = df_prepared[col].astype(int)
            if df_prepared[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_prepared[col]):
                    df_prepared[col].fillna(df_prepared[col].median(), inplace=True)
                else:
                    df_prepared[col].fillna('Unknown', inplace=True)
        return df_prepared

    def _detect_discrete_columns(self, df: pd.DataFrame) -> List[str]:
        discrete_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                discrete_columns.append(col)
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_ratio = df[col].nunique() / max(1, len(df))
                if unique_ratio < 0.05:
                    discrete_columns.append(col)
        return discrete_columns

    def _post_process_ctgan_output(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        for col in synthetic_df.columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                min_val = original_df[col].min()
                max_val = original_df[col].max()
                synthetic_df[col] = synthetic_df[col].clip(min_val, max_val)
                if pd.api.types.is_integer_dtype(original_df[col]):
                    synthetic_df[col] = synthetic_df[col].round().astype(original_df[col].dtype)
            elif synthetic_df[col].dtype == 'object':
                original_categories = set(original_df[col].dropna().astype(str).unique())
                def map_to_original(val):
                    if pd.isna(val):
                        return val
                    v = str(val)
                    if v in original_categories:
                        return v
                    if original_categories:
                        return np.random.choice(list(original_categories))
                    return v
                synthetic_df[col] = synthetic_df[col].apply(map_to_original)
        return synthetic_df

    def _generate_statistical(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        logger.info(f"📊 Statistical fallback for {num_rows} rows")
        out = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                out[c] = self._gen_num(df[c], num_rows)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                out[c] = self._gen_dt(df[c], num_rows)
            else:
                out[c] = self._gen_cat(df[c], num_rows)
        return pd.DataFrame(out)

    def _gen_num(self, s: pd.Series, n: int) -> np.ndarray:
        clean = s.dropna()
        if len(clean) == 0:
            return np.zeros(n)
        if pd.api.types.is_integer_dtype(s):
            mean, std = clean.mean(), clean.std()
            std = 1 if (pd.isna(std) or std == 0) else std
            x = np.random.normal(mean, std, n)
            x = np.round(x).astype(int)
            return np.clip(x, clean.min(), clean.max())
        else:
            mean, std = clean.mean(), clean.std()
            std = 1.0 if (pd.isna(std) or std == 0) else std
            x = np.random.normal(mean, std, n)
            return np.clip(x, clean.min(), clean.max())

    def _gen_cat(self, s: pd.Series, n: int) -> np.ndarray:
        clean = s.dropna().astype(str)
        if len(clean) == 0:
            return np.array(['Unknown'] * n)
        vc = clean.value_counts()
        cats = vc.index.tolist()
        probs = (vc / len(clean)).tolist()
        return np.random.choice(cats, size=n, p=probs)

    def _gen_dt(self, s: pd.Series, n: int) -> np.ndarray:
        clean = s.dropna()
        if len(clean) == 0:
            return np.array([pd.Timestamp.now()] * n)
        return np.random.choice(clean.values, size=n, replace=True)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Provide concise info for quality reports and logging.
        Matches callers in multi_table_processor.py expecting this method.
        """
        return {
            'type': 'CTGAN' if (self.use_ctgan and getattr(self, 'ctgan_available', False)) else 'Statistical',
            'ctgan_available': getattr(self, 'ctgan_available', False),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'max_train_seconds': self.max_train_seconds
        }
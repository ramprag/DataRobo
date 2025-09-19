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

            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error with {encoding} encoding: {e}")
                    continue

            if df is None:
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

        try:
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
                if analysis[column]['is_numeric'] and not col_data.dropna().empty:
                    try:
                        analysis[column].update({
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std())
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating numeric stats for {column}: {e}")

        except Exception as e:
            logger.error(f"Error analyzing data types: {e}")

        return analysis

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        try:
            df_clean = df.copy()

            # Remove completely empty columns
            df_clean = df_clean.dropna(axis=1, how='all')

            logger.info("Data cleaning completed")
            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return metrics"""
        try:
            metrics = {
                'row_count': len(df),
                'total_columns': len(df.columns),
                'missing_data_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 and len(df.columns) > 0 else 0,
                'duplicate_rows': int(df.duplicated().sum()),
                'completely_null_columns': int((df.isnull().all()).sum()),
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
            if metrics['row_count'] > 0:
                duplicate_percentage = metrics['duplicate_rows'] / metrics['row_count'] * 100
                if duplicate_percentage > 20:
                    quality_score -= 20
                elif duplicate_percentage > 5:
                    quality_score -= 10

            # Penalize for completely null columns
            if metrics['completely_null_columns'] > 0:
                quality_score -= metrics['completely_null_columns'] * 5

            metrics['quality_score'] = max(0, quality_score)

            return metrics

        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {
                'row_count': len(df) if df is not None else 0,
                'total_columns': len(df.columns) if df is not None else 0,
                'quality_score': 50.0,
                'error': str(e)
            }

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
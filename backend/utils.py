import os
import re
import logging
import mimetypes
from typing import List, Optional, Dict, Any
from pathlib import Path
import unicodedata

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/synthetic_data_generator.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")

    return logger

def validate_file_type(filename: str) -> bool:
    """Validate if the uploaded file type is supported"""

    if not filename:
        return False

    # Get file extension
    file_extension = Path(filename).suffix.lower()

    # Supported file types
    supported_extensions = {'.csv', '.xlsx', '.xls', '.json'}

    return file_extension in supported_extensions

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues"""

    if not filename:
        return "unknown_file"

    # Remove directory path components
    filename = os.path.basename(filename)

    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)

    # Remove or replace problematic characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)

    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)

    # Ensure filename is not empty and has reasonable length
    if not filename or filename.startswith('.'):
        filename = 'file_' + filename

    # Limit filename length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext

    return filename

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""

    if not os.path.exists(file_path):
        return {'error': 'File does not exist'}

    stat = os.stat(file_path)

    file_info = {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified_time': stat.st_mtime,
        'mime_type': mimetypes.guess_type(file_path)[0],
        'extension': Path(file_path).suffix.lower()
    }

    return file_info

def validate_csv_structure(file_path: str) -> Dict[str, Any]:
    """Validate CSV file structure and provide basic analysis"""

    try:
        import pandas as pd

        # Try to read first few rows to validate structure
        df_sample = pd.read_csv(file_path, nrows=10)

        validation = {
            'is_valid': True,
            'columns': df_sample.columns.tolist(),
            'column_count': len(df_sample.columns),
            'sample_rows': len(df_sample),
            'has_headers': True,  # Assume headers by default
            'empty_columns': [],
            'data_types': df_sample.dtypes.to_dict()
        }

        # Check for empty columns in sample
        for col in df_sample.columns:
            if df_sample[col].isnull().all():
                validation['empty_columns'].append(col)

        return validation

    except Exception as e:
        return {
            'is_valid': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def create_safe_directory(directory_path: str) -> bool:
    """Safely create directory with proper permissions"""

    try:
        os.makedirs(directory_path, exist_ok=True)

        # Set appropriate permissions (read/write for owner, read for group)
        os.chmod(directory_path, 0o755)

        return True

    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        return False

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """Cleanup old files from a directory"""

    if not os.path.exists(directory):
        return 0

    import time

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    files_removed = 0

    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > max_age_seconds:
                    os.remove(file_path)
                    files_removed += 1
                    logging.info(f"Removed old file: {file_path}")

        return files_removed

    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        return 0

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def validate_privacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate privacy configuration"""

    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    required_fields = ['mask_emails', 'mask_names', 'mask_phone_numbers', 'anonymization_method']

    for field in required_fields:
        if field not in config:
            validation_result['errors'].append(f"Missing required field: {field}")
            validation_result['is_valid'] = False

    # Validate anonymization method
    valid_methods = ['faker', 'hash', 'redact']
    if 'anonymization_method' in config:
        if config['anonymization_method'] not in valid_methods:
            validation_result['errors'].append(f"Invalid anonymization method. Must be one of: {valid_methods}")
            validation_result['is_valid'] = False

    # Validate boolean fields
    boolean_fields = ['mask_emails', 'mask_names', 'mask_phone_numbers', 'mask_addresses', 'mask_ssn']
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            validation_result['warnings'].append(f"Field {field} should be boolean")

    # Validate custom fields
    if 'custom_fields' in config:
        if not isinstance(config['custom_fields'], list):
            validation_result['errors'].append("custom_fields must be a list")
            validation_result['is_valid'] = False

    return validation_result

def calculate_processing_estimate(file_size_mb: float, num_rows: int = None) -> Dict[str, Any]:
    """Estimate processing time and resources needed"""

    # Base estimates (rough approximations)
    base_time_per_mb = 5  # seconds per MB
    base_memory_per_mb = 3  # MB of RAM per MB of data

    # Adjust based on number of rows if known
    if num_rows:
        if num_rows > 100000:
            base_time_per_mb *= 2  # More complex processing for large datasets
        elif num_rows < 1000:
            base_time_per_mb *= 0.5  # Faster for small datasets

    estimate = {
        'estimated_time_seconds': int(file_size_mb * base_time_per_mb),
        'estimated_memory_mb': int(file_size_mb * base_memory_per_mb),
        'complexity_level': 'low' if file_size_mb < 10 else 'medium' if file_size_mb < 100 else 'high'
    }

    # Format human-readable time
    seconds = estimate['estimated_time_seconds']
    if seconds < 60:
        estimate['estimated_time_human'] = f"{seconds} seconds"
    elif seconds < 3600:
        estimate['estimated_time_human'] = f"{seconds // 60} minutes"
    else:
        estimate['estimated_time_human'] = f"{seconds // 3600} hours"

    return estimate

def generate_dataset_summary(df) -> Dict[str, Any]:
    """Generate a comprehensive summary of a dataset"""

    import pandas as pd
    import numpy as np

    summary = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'data_types': df.dtypes.value_counts().to_dict()
        },
        'data_quality': {
            'missing_values_total': int(df.isnull().sum().sum()),
            'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2),
            'completely_null_columns': list(df.columns[df.isnull().all()]),
            'columns_with_nulls': list(df.columns[df.isnull().any()])
        },
        'column_analysis': {},
        'statistical_summary': {}
    }

    # Analyze each column
    for column in df.columns:
        col_data = df[column]
        col_analysis = {
            'data_type': str(col_data.dtype),
            'null_count': int(col_data.isnull().sum()),
            'null_percentage': round(col_data.isnull().sum() / len(col_data) * 100, 2),
            'unique_values': int(col_data.nunique()),
            'unique_percentage': round(col_data.nunique() / len(col_data) * 100, 2)
        }

        # Type-specific analysis
        if pd.api.types.is_numeric_dtype(col_data):
            col_analysis.update({
                'min': float(col_data.min()) if not col_data.empty else None,
                'max': float(col_data.max()) if not col_data.empty else None,
                'mean': float(col_data.mean()) if not col_data.empty else None,
                'median': float(col_data.median()) if not col_data.empty else None,
                'std': float(col_data.std()) if not col_data.empty else None
            })
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            # Get top categories
            top_categories = col_data.value_counts().head(5)
            col_analysis['top_categories'] = top_categories.to_dict()

        summary['column_analysis'][column] = col_analysis

    # Overall statistical summary for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        summary['statistical_summary'] = {
            'numeric_columns_count': len(numeric_columns),
            'correlation_matrix_available': len(numeric_columns) > 1,
            'has_outliers': False  # Placeholder for outlier detection
        }

        # Simple outlier detection using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                summary['statistical_summary']['has_outliers'] = True
                break

    return summary

def validate_synthetic_data_requirements(original_df, synthetic_df) -> Dict[str, Any]:
    """Validate that synthetic data meets basic requirements"""

    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'checks_passed': 0,
        'total_checks': 0
    }

    checks = [
        ('column_names_match', lambda: set(original_df.columns) == set(synthetic_df.columns)),
        ('column_count_match', lambda: len(original_df.columns) == len(synthetic_df.columns)),
        ('has_data', lambda: len(synthetic_df) > 0),
        ('no_all_null_columns', lambda: not synthetic_df.isnull().all().any()),
        ('data_types_reasonable', lambda: all(
            str(original_df[col].dtype) == str(synthetic_df[col].dtype)
            or (pd.api.types.is_numeric_dtype(original_df[col]) and pd.api.types.is_numeric_dtype(synthetic_df[col]))
            for col in original_df.columns if col in synthetic_df.columns
        ))
    ]

    validation['total_checks'] = len(checks)

    for check_name, check_func in checks:
        try:
            if check_func():
                validation['checks_passed'] += 1
            else:
                validation['errors'].append(f"Check failed: {check_name}")
                validation['is_valid'] = False
        except Exception as e:
            validation['errors'].append(f"Check error ({check_name}): {str(e)}")
            validation['is_valid'] = False

    # Additional warnings
    if len(synthetic_df) < len(original_df) * 0.5:
        validation['warnings'].append("Synthetic dataset is significantly smaller than original")

    if len(synthetic_df) > len(original_df) * 2:
        validation['warnings'].append("Synthetic dataset is significantly larger than original")

    return validation

class ProgressTracker:
    """Track progress of long-running operations"""

    def __init__(self, total_steps: int, operation_name: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = None
        self.step_times = []
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start tracking progress"""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name} with {self.total_steps} steps")

    def update(self, step_name: str = ""):
        """Update progress by one step"""
        import time

        current_time = time.time()
        if self.current_step > 0:
            step_duration = current_time - self.step_times[-1] if self.step_times else 0
            self.step_times.append(current_time)
        else:
            self.step_times.append(current_time)

        self.current_step += 1
        progress_percent = (self.current_step / self.total_steps) * 100

        # Estimate remaining time
        if len(self.step_times) > 1:
            avg_step_time = (current_time - self.start_time) / self.current_step
            remaining_time = avg_step_time * (self.total_steps - self.current_step)
            remaining_str = f" (ETA: {int(remaining_time)}s)"
        else:
            remaining_str = ""

        step_info = f" - {step_name}" if step_name else ""
        self.logger.info(f"{self.operation_name}: {progress_percent:.1f}% complete{step_info}{remaining_str}")

    def complete(self):
        """Mark operation as complete"""
        import time

        total_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"{self.operation_name} completed in {total_time:.1f} seconds")

def safe_json_serialize(obj) -> str:
    """Safely serialize objects to JSON, handling numpy types and dates"""

    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime, date

    def json_serializer(obj):
        """Custom JSON serializer for numpy and pandas objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        logging.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": "Serialization failed", "details": str(e)})

def validate_environment() -> Dict[str, Any]:
    """Validate the environment and dependencies"""

    validation = {
        'python_version': None,
        'required_packages': {},
        'optional_packages': {},
        'system_info': {},
        'recommendations': []
    }

    # Check Python version
    import sys
    validation['python_version'] = sys.version

    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'fastapi', 'uvicorn', 'sqlalchemy',
        'pydantic', 'faker', 'scipy'
    ]

    for package in required_packages:
        try:
            module = __import__(package)
            validation['required_packages'][package] = {
                'installed': True,
                'version': getattr(module, '__version__', 'unknown')
            }
        except ImportError:
            validation['required_packages'][package] = {
                'installed': False,
                'version': None
            }

    # Check optional packages
    optional_packages = ['sdv', 'plotly', 'seaborn', 'matplotlib']

    for package in optional_packages:
        try:
            module = __import__(package)
            validation['optional_packages'][package] = {
                'installed': True,
                'version': getattr(module, '__version__', 'unknown')
            }
        except ImportError:
            validation['optional_packages'][package] = {
                'installed': False,
                'version': None
            }

    # System info
    validation['system_info'] = {
        'platform': sys.platform,
        'python_executable': sys.executable
    }

    # Generate recommendations
    missing_required = [pkg for pkg, info in validation['required_packages'].items() if not info['installed']]
    if missing_required:
        validation['recommendations'].append(f"Install missing required packages: {', '.join(missing_required)}")

    missing_optional = [pkg for pkg, info in validation['optional_packages'].items() if not info['installed']]
    if missing_optional:
        validation['recommendations'].append(f"Consider installing optional packages for enhanced functionality: {', '.join(missing_optional)}")

    return validation
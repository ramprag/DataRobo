"""
Enhanced Multi-Table Processor with GAN Support
Complete replacement for multi_table_processor.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import zipfile
import io
import json
from pathlib import Path
import hashlib
import uuid
from datetime import datetime
import logging

# Import existing classes
from privacy_masker import PrivacyMasker
from quality_validator import QualityValidator

# Import the new GAN generator
from gan_synthetic_generator import GANSyntheticDataGenerator

logger = logging.getLogger(__name__)

class TableRelationshipDetector:
    """Detects and manages relationships between tables"""

    def __init__(self):
        self.relationships = {}
        self.primary_keys = {}
        self.foreign_keys = {}
        self.table_schemas = {}

    def analyze_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze multiple tables to detect relationships"""
        logger.info(f"Analyzing {len(tables)} tables for relationships")

        # Store schemas
        for table_name, df in tables.items():
            self.table_schemas[table_name] = {
                'columns': list(df.columns),
                'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'row_count': len(df),
                'null_counts': df.isnull().sum().to_dict()
            }

        # Detect primary keys
        for table_name, df in tables.items():
            self.primary_keys[table_name] = self._detect_primary_key(df, table_name)

        # Detect foreign keys
        self._detect_foreign_keys(tables)

        # Get generation order
        dependency_order = self._get_generation_order()

        logger.info(f"Detected {len(self.relationships)} relationships")
        logger.info(f"Primary keys: {self.primary_keys}")
        logger.info(f"Foreign keys: {self.foreign_keys}")

        return {
            'relationships': self.relationships,
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys,
            'schemas': self.table_schemas,
            'generation_order': dependency_order
        }

    def _detect_primary_key(self, df: pd.DataFrame, table_name: str) -> Optional[str]:
        """Detect primary key column"""
        candidates = []

        for col in df.columns:
            col_data = df[col].dropna()

            if len(col_data) < len(df) * 0.95:
                continue

            if col_data.nunique() == len(col_data):
                score = 0.0
                col_lower = col.lower()

                if col_lower == 'id':
                    score += 10.0
                elif col_lower.endswith('_id') or col_lower.endswith('id'):
                    score += 8.0
                elif 'key' in col_lower:
                    score += 5.0
                elif col_lower.startswith('id_'):
                    score += 6.0

                if pd.api.types.is_integer_dtype(df[col]):
                    score += 3.0

                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        values = sorted(col_data.values)
                        if values == list(range(min(values), max(values) + 1)):
                            score += 5.0
                    except:
                        pass

                candidates.append({'column': col, 'score': score})

        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            pk = candidates[0]['column']
            logger.info(f"✓ Detected primary key '{pk}' for table '{table_name}' (score: {candidates[0]['score']})")
            return pk

        logger.warning(f"✗ No primary key detected for table '{table_name}'")
        return None

    def _detect_foreign_keys(self, tables: Dict[str, pd.DataFrame]):
        """Detect foreign key relationships"""
        table_names = list(tables.keys())

        for parent_table in table_names:
            parent_pk = self.primary_keys.get(parent_table)
            if not parent_pk:
                continue

            parent_df = tables[parent_table]
            parent_values = set(parent_df[parent_pk].dropna().values)

            if len(parent_values) == 0:
                continue

            for child_table in table_names:
                if parent_table == child_table:
                    continue

                child_df = tables[child_table]

                for col in child_df.columns:
                    if self._is_foreign_key_candidate(
                        parent_table, parent_pk, parent_values,
                        child_table, col, child_df
                    ):
                        child_values = set(child_df[col].dropna().values)
                        overlap = len(child_values.intersection(parent_values))

                        if len(child_values) > 0:
                            confidence = overlap / len(child_values)

                            if confidence >= 0.7:
                                self._add_relationship(
                                    parent_table, parent_pk,
                                    child_table, col, confidence
                                )

    def _is_foreign_key_candidate(self, parent_table: str, parent_pk: str,
                                  parent_values: set, child_table: str,
                                  child_col: str, child_df: pd.DataFrame) -> bool:
        """Check if a column is a foreign key candidate"""

        if child_col == parent_pk and parent_table == child_table:
            return False

        col_lower = child_col.lower()
        parent_lower = parent_table.lower()
        pk_lower = parent_pk.lower()

        name_matches = [
            col_lower == pk_lower,
            col_lower == f"{parent_lower}_{pk_lower}",
            col_lower == f"{parent_lower}{pk_lower}",
            col_lower.endswith(f"_{pk_lower}"),
            col_lower.startswith(f"{parent_lower}_"),
            parent_lower in col_lower and 'id' in col_lower,
        ]

        if not any(name_matches):
            return False

        parent_dtype = str(type(list(parent_values)[0]).__name__) if parent_values else None
        child_dtype = str(child_df[child_col].dtype)

        if not self._are_types_compatible(parent_dtype, child_dtype):
            return False

        return True

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible"""
        numeric_types = ['int', 'int32', 'int64', 'float', 'float32', 'float64']
        string_types = ['object', 'str', 'string']

        if any(t in type1.lower() for t in numeric_types):
            return any(t in type2.lower() for t in numeric_types)
        elif any(t in type1.lower() for t in string_types):
            return any(t in type2.lower() for t in string_types)

        return True

    def _add_relationship(self, parent_table: str, parent_col: str,
                         child_table: str, child_col: str, confidence: float):
        """Add detected relationship"""
        rel_key = f"{parent_table}.{parent_col} -> {child_table}.{child_col}"

        self.relationships[rel_key] = {
            'parent_table': parent_table,
            'parent_column': parent_col,
            'child_table': child_table,
            'child_column': child_col,
            'confidence': confidence,
            'type': 'foreign_key'
        }

        if child_table not in self.foreign_keys:
            self.foreign_keys[child_table] = []

        self.foreign_keys[child_table].append({
            'column': child_col,
            'references_table': parent_table,
            'references_column': parent_col,
            'confidence': confidence
        })

        logger.info(f"✓ Detected FK: {rel_key} (confidence: {confidence:.2%})")

    def _get_generation_order(self) -> List[str]:
        """Get topological order for table generation"""
        all_tables = set(self.table_schemas.keys())
        visited = set()
        order = []

        graph = {table: [] for table in all_tables}
        for rel in self.relationships.values():
            if rel['type'] == 'foreign_key':
                graph[rel['child_table']].append(rel['parent_table'])

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in graph.get(node, []):
                if parent not in visited:
                    visit(parent)
            order.append(node)

        for node in all_tables:
            if node not in visited:
                visit(node)

        logger.info(f"Generation order: {order}")
        return order


class SyntheticKeyManager:
    """Manages mapping between original and synthetic keys"""

    def __init__(self):
        self.key_mappings = {}

    def create_mapping(self, table_name: str, original_values: pd.Series) -> Dict:
        """Create mapping from original to synthetic keys"""
        mapping = {}
        unique_values = original_values.dropna().unique()

        for i, original_key in enumerate(unique_values):
            synthetic_key = self._generate_synthetic_key(table_name, original_key, i)
            mapping[original_key] = synthetic_key

        mapping_id = str(uuid.uuid4())
        self.key_mappings[mapping_id] = {
            'table': table_name,
            'mapping': mapping,
            'created_at': datetime.now()
        }

        logger.info(f"Created key mapping for {table_name}: {len(mapping)} unique values")
        return {'mapping_id': mapping_id, 'mapping': mapping}

    def _generate_synthetic_key(self, table_name: str, original_key: Any, index: int) -> Any:
        """Generate synthetic key maintaining data type"""
        hash_input = f"{table_name}_{original_key}_{index}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()[:8]

        if isinstance(original_key, (int, np.integer)):
            return int(hash_value, 16) % 1000000
        elif isinstance(original_key, str):
            return f"syn_{hash_value}"
        else:
            return hash_value

    def apply_mapping(self, values: pd.Series, mapping_id: str) -> pd.Series:
        """Apply mapping to transform keys"""
        if mapping_id not in self.key_mappings:
            raise ValueError(f"Mapping {mapping_id} not found")

        mapping = self.key_mappings[mapping_id]['mapping']
        return values.map(mapping).fillna(values)


class EnhancedSyntheticDataGenerator:
    """Multi-table synthetic data generator with GAN support"""

    def __init__(self, use_gan: bool = True, gan_epochs: int = 300):
        """
        Initialize generator with GAN support

        Args:
            use_gan: Whether to use GAN (CTGAN) for generation
            gan_epochs: Number of epochs for GAN training
        """
        self.use_gan = use_gan
        self.gan_generator = GANSyntheticDataGenerator(
            use_ctgan=use_gan,
            epochs=gan_epochs,
            batch_size=500,
            max_train_seconds=300  # Give CTGAN more headroom; adaptive cap still applies
        )
        self.privacy_masker = PrivacyMasker()
        self.quality_validator = QualityValidator()
        self.relationship_detector = TableRelationshipDetector()
        self.key_manager = SyntheticKeyManager()

    def process_upload(self, file_data: bytes, filename: str, privacy_config, num_rows: Optional[int] = None) -> Dict:
        """Process upload - handles both single and multiple tables"""
        logger.info(f"Processing upload: {filename}")
        logger.info(f"GAN Mode: {'ENABLED' if self.use_gan else 'DISABLED'}")
        logger.info(f"Target rows: {num_rows if num_rows else 'Same as original'}")

        tables = self._extract_tables(file_data, filename)
        logger.info(f"Extracted {len(tables)} table(s): {list(tables.keys())}")

        if len(tables) == 1:
            table_name = list(tables.keys())[0]
            df = list(tables.values())[0]
            return self._process_single_table(df, table_name, privacy_config, num_rows)
        else:
            return self._process_multiple_tables(tables, privacy_config, num_rows)

    def _process_single_table(self, df: pd.DataFrame, filename: str, privacy_config, num_rows: Optional[int] = None) -> Dict:
        """Process single table using GAN"""
        logger.info(f"Processing single table: {filename}")
        logger.info(f"Using {'GAN (CTGAN)' if self.use_gan else 'Statistical'} method")
        logger.info(f"Original rows: {len(df)}, Target rows: {num_rows if num_rows else len(df)}")

        # Apply privacy masks
        masked_df = self.privacy_masker.apply_privacy_masks(df, privacy_config)

        # Generate synthetic data using GAN with custom row count
        synthetic_df = self.gan_generator.generate_synthetic_data(
            masked_df,
            num_rows=num_rows,  # Use the custom num_rows parameter
            preserve_columns=[]
        )

        # Validate quality
        quality_metrics = self.quality_validator.compare_distributions(df, synthetic_df)

        # Add model info to quality metrics
        quality_metrics['generation_method'] = self.gan_generator.get_model_info()

        return {
            'status': 'completed',
            'table_count': 1,
            'tables': {Path(filename).stem: synthetic_df},
            'relationships': {},
            'quality_metrics': quality_metrics,
            'relationship_summary': {
                'total_relationships': 0,
                'foreign_keys': [],
                'primary_keys': []
            }
        }

    def _process_multiple_tables(self, tables: Dict[str, pd.DataFrame], privacy_config, num_rows: Optional[int] = None) -> Dict:
        """Process multiple tables with relationship detection"""
        logger.info(f"Processing {len(tables)} tables with relationships")
        logger.info(f"Target rows per table: {num_rows if num_rows else 'Same as original'}")

        # Detect relationships
        relationships = self._detect_relationships(tables)
        logger.info(f"Detected {len(relationships)} relationship(s)")

        # Apply privacy masks to all tables
        masked_tables = {}
        for table_name, df in tables.items():
            masked_tables[table_name] = self.privacy_masker.apply_privacy_masks(df, privacy_config)

        # Generate synthetic data for each table
        synthetic_tables = {}
        quality_metrics = {}

        for table_name, masked_df in masked_tables.items():
            logger.info(f"Generating synthetic data for {table_name}")

            # Use custom num_rows for each table
            synthetic_df = self.gan_generator.generate_synthetic_data(
                masked_df,
                num_rows=num_rows,  # Use the custom num_rows parameter
                preserve_columns=[]
            )

            synthetic_tables[table_name] = synthetic_df

            # Validate quality for this table
            original_df = tables[table_name]
            quality_metrics[table_name] = self.quality_validator.compare_distributions(original_df, synthetic_df)

        # Add model info to quality metrics
        for table_name in quality_metrics:
            quality_metrics[table_name]['generation_method'] = self.gan_generator.get_model_info()

        return {
            'status': 'completed',
            'table_count': len(tables),
            'tables': synthetic_tables,
            'relationships': relationships,
            'quality_metrics': quality_metrics,
            'relationship_summary': self._create_relationship_summary(relationships, tables)
        }

    def _extract_tables(self, file_data: bytes, filename: str) -> Dict[str, pd.DataFrame]:
        """Extract tables from uploaded file(s)"""
        tables = {}

        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_data)) as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.lower().endswith(('.csv', '.xlsx', '.xls')) and not file_info.filename.startswith('__MACOSX'):
                        try:
                            with zip_file.open(file_info) as f:
                                table_name = Path(file_info.filename).stem
                                tables[table_name] = self._read_table_file(f.read(), file_info.filename)
                                logger.info(f"Extracted table '{table_name}' with {len(tables[table_name])} rows")
                        except Exception as e:
                            logger.warning(f"Failed to extract {file_info.filename}: {e}")
        elif filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            table_name = Path(filename).stem
            tables[table_name] = self._read_table_file(file_data, filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        return tables

    def _read_table_file(self, file_data: bytes, filename: str) -> pd.DataFrame:
        """Read individual table file"""
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_data))
        elif filename.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(file_data))
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    # ADD THESE MISSING METHODS FOR COMPATIBILITY WITH MAIN.PY
    def _detect_relationships(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """
        Detect relationships between tables (compatibility method for main.py)
        Returns format expected by main.py upload endpoint
        """
        logger.info("Running relationship detection for upload analysis")

        # Use the relationship detector
        relationship_info = self.relationship_detector.analyze_tables(tables)

        # Convert to format expected by main.py
        relationships = {}
        for table_name, fks in relationship_info['foreign_keys'].items():
            relationships[table_name] = []
            for fk in fks:
                relationships[table_name].append({
                    'foreign_key': fk['column'],
                    'referenced_table': fk['references_table'],
                    'referenced_key': fk['references_column'],
                    'confidence': fk['confidence']
                })

        logger.info(f"Detected {sum(len(v) for v in relationships.values())} relationships")
        return relationships

    def _create_relationship_summary(
        self,
        relationships: Dict[str, List[Dict]],
        tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Create relationship summary (compatibility method for main.py)
        Returns format expected by main.py upload endpoint
        """
        logger.info("Creating relationship summary for upload response")

        # Run full analysis to get all info
        relationship_info = self.relationship_detector.analyze_tables(tables)

        # Build summary in expected format
        tables_with_pks = set()
        tables_with_fks = set(relationships.keys())

        # Identify which tables are referenced (have PKs)
        for table_rels in relationships.values():
            for rel in table_rels:
                tables_with_pks.add(rel['referenced_table'])

        # Create detailed relationship descriptions
        relationship_details = []
        for table_name, table_rels in relationships.items():
            for rel in table_rels:
                description = (
                    f"{table_name}.{rel['foreign_key']} → "
                    f"{rel['referenced_table']}.{rel['referenced_key']}"
                )
                relationship_details.append({
                    'description': description,
                    'from_table': table_name,
                    'to_table': rel['referenced_table'],
                    'confidence': rel['confidence']
                })

        summary = {
            'total_relationships': len(relationship_details),
            'tables_with_primary_keys': len(tables_with_pks),
            'tables_with_foreign_keys': len(tables_with_fks),
            'generation_order': relationship_info['generation_order'],
            'relationship_details': relationship_details
        }

        logger.info(f"Relationship summary: {summary['total_relationships']} relationships found")
        return summary


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

# Import your existing classes
from privacy_masker import PrivacyMasker
from synthetic_generator import SyntheticDataGenerator
from quality_validator import QualityValidator

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

        for table_name, df in tables.items():
            self.table_schemas[table_name] = {
                'columns': list(df.columns),
                'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'row_count': len(df),
                'null_counts': df.isnull().sum().to_dict()
            }

        for table_name, df in tables.items():
            self.primary_keys[table_name] = self._detect_primary_key(df, table_name)

        self._detect_foreign_keys(tables)
        dependency_order = self._get_generation_order()

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
            if df[col].nunique() == len(df) and df[col].isnull().sum() == 0:
                score = 0.0
                if any(kw in col.lower() for kw in ['id', '_id', 'key', '_key']):
                    score += 5.0
                if pd.api.types.is_integer_dtype(df[col]):
                    score += 3.0
                candidates.append({'column': col, 'score': score})

        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            pk = candidates[0]['column']
            logger.info(f"Detected primary key '{pk}' for table '{table_name}'")
            return pk
        return None

    def _detect_foreign_keys(self, tables: Dict[str, pd.DataFrame]):
        """Detect foreign key relationships between tables"""
        table_names = list(tables.keys())

        for i, parent_table in enumerate(table_names):
            for j, child_table in enumerate(table_names):
                if i != j:
                    self._find_fk_relationship(
                        parent_table, tables[parent_table],
                        child_table, tables[child_table]
                    )

    def _find_fk_relationship(self, parent_name: str, parent_df: pd.DataFrame,
                             child_name: str, child_df: pd.DataFrame):
        """Find foreign key relationship between two tables"""
        parent_pk = self.primary_keys.get(parent_name)
        if not parent_pk:
            return

        parent_values = set(parent_df[parent_pk].dropna())

        for col in child_df.columns:
            if col == parent_pk or f"{parent_name}_{parent_pk}".lower() in col.lower():
                child_values = set(child_df[col].dropna())

                if len(child_values) > 0:
                    overlap = len(child_values.intersection(parent_values))
                    confidence = overlap / len(child_values)

                    if confidence > 0.8:
                        self._add_relationship(parent_name, parent_pk, child_name, col, confidence)

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

        logger.info(f"Detected FK: {rel_key} (confidence: {confidence:.2f})")

    def _get_generation_order(self) -> List[str]:
        """Get topological order for table generation"""
        visited = set()
        order = []

        graph = {table: [] for table in self.table_schemas.keys()}
        for rel in self.relationships.values():
            if rel['type'] == 'foreign_key':
                graph[rel['parent_table']].append(rel['child_table'])

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for child in graph.get(node, []):
                if child not in visited:
                    visit(child)
            order.insert(0, node)

        for node in graph.keys():
            visit(node)

        return order


class SyntheticKeyManager:
    """Manages mapping between original and synthetic keys"""

    def __init__(self):
        self.key_mappings = {}

    def create_mapping(self, table_name: str, original_values: pd.Series) -> Dict:
        """Create mapping from original to synthetic keys"""
        mapping = {}

        for i, original_key in enumerate(original_values.unique()):
            synthetic_key = self._generate_synthetic_key(table_name, original_key, i)
            mapping[original_key] = synthetic_key

        mapping_id = str(uuid.uuid4())
        self.key_mappings[mapping_id] = {
            'table': table_name,
            'mapping': mapping,
            'created_at': datetime.now()
        }

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
        """Apply mapping to transform original keys to synthetic"""
        if mapping_id not in self.key_mappings:
            raise ValueError(f"Mapping {mapping_id} not found")

        mapping = self.key_mappings[mapping_id]['mapping']
        return values.map(mapping).fillna(values)


class EnhancedSyntheticDataGenerator:
    """Multi-table synthetic data generator with relationship preservation"""

    def __init__(self):
        self.single_table_generator = SyntheticDataGenerator()
        self.privacy_masker = PrivacyMasker()
        self.quality_validator = QualityValidator()
        self.relationship_detector = TableRelationshipDetector()
        self.key_manager = SyntheticKeyManager()

    def process_upload(self, file_data: bytes, filename: str, privacy_config) -> Dict:
        """Process upload - handles both single and multiple tables"""
        logger.info(f"Processing upload: {filename}")

        tables = self._extract_tables(file_data, filename)

        if len(tables) == 1:
            table_name = list(tables.keys())[0]
            df = list(tables.values())[0]
            return self._process_single_table(df, table_name, privacy_config)
        else:
            return self._process_multiple_tables(tables, privacy_config)

    def _extract_tables(self, file_data: bytes, filename: str) -> Dict[str, pd.DataFrame]:
        """Extract tables from uploaded file(s)"""
        tables = {}

        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_data)) as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                        with zip_file.open(file_info) as f:
                            table_name = Path(file_info.filename).stem
                            tables[table_name] = self._read_table_file(f.read(), file_info.filename)
        elif filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            table_name = Path(filename).stem
            tables[table_name] = self._read_table_file(file_data, filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def _read_table_file(self, file_data: bytes, filename: str) -> pd.DataFrame:
        """Read individual table file"""
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_data))
        elif filename.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(file_data))
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def _process_single_table(self, df: pd.DataFrame, filename: str, privacy_config) -> Dict:
        """Process single table using existing logic"""
        masked_df = self.privacy_masker.apply_privacy_masks(df, privacy_config)
        synthetic_df = self.single_table_generator.generate_synthetic_data(masked_df)
        quality_metrics = self.quality_validator.compare_distributions(df, synthetic_df)

        return {
            'status': 'completed',
            'table_count': 1,
            'tables': {Path(filename).stem: synthetic_df},
            'relationships': {},
            'quality_metrics': quality_metrics,
            'relationship_summary': {
                'total_relationships': 0,
                'tables_with_primary_keys': 0,
                'tables_with_foreign_keys': 0,
                'generation_order': [Path(filename).stem],
                'relationship_details': []
            }
        }

    def _process_multiple_tables(self, tables: Dict[str, pd.DataFrame], privacy_config) -> Dict:
        """Process multiple related tables"""
        logger.info("Starting multi-table processing")

        relationship_info = self.relationship_detector.analyze_tables(tables)

        synthetic_tables = {}
        key_mappings = {}

        for table_name in relationship_info['generation_order']:
            logger.info(f"Generating synthetic data for table: {table_name}")

            original_df = tables[table_name]
            masked_df = self.privacy_masker.apply_privacy_masks(original_df, privacy_config)

            synthetic_df, mapping_info = self._generate_table_with_relationships(
                table_name, masked_df, relationship_info, key_mappings
            )

            synthetic_tables[table_name] = synthetic_df
            if mapping_info:
                key_mappings.update(mapping_info)

        original_dfs = list(tables.values())
        synthetic_dfs = list(synthetic_tables.values())

        quality_metrics = self.quality_validator.compare_distributions(
            original_dfs[0], synthetic_dfs[0]
        ) if len(original_dfs) > 0 else {'overall_quality_score': 50.0}

        return {
            'status': 'completed',
            'table_count': len(tables),
            'tables': synthetic_tables,
            'relationships': relationship_info['relationships'],
            'quality_metrics': quality_metrics,
            'relationship_summary': self._summarize_relationships(relationship_info)
        }

    def _generate_table_with_relationships(self, table_name: str, df: pd.DataFrame,
                                         relationship_info: Dict, key_mappings: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic table while preserving relationships"""
        synthetic_df = df.copy()
        new_mappings = {}

        pk_column = relationship_info['primary_keys'].get(table_name)
        if pk_column and pk_column in df.columns:
            mapping_result = self.key_manager.create_mapping(table_name, df[pk_column])
            synthetic_df[pk_column] = self.key_manager.apply_mapping(
                df[pk_column], mapping_result['mapping_id']
            )
            new_mappings[f"{table_name}_{pk_column}"] = mapping_result['mapping_id']

        fks = relationship_info['foreign_keys'].get(table_name, [])
        for fk in fks:
            fk_column = fk['column']
            parent_table = fk['references_table']
            parent_pk = fk['references_column']

            parent_mapping_key = f"{parent_table}_{parent_pk}"
            if parent_mapping_key in key_mappings:
                synthetic_df[fk_column] = self.key_manager.apply_mapping(
                    df[fk_column], key_mappings[parent_mapping_key]
                )

        preserve_columns = []
        if pk_column:
            preserve_columns.append(pk_column)
        preserve_columns.extend([fk['column'] for fk in fks])

        for column in synthetic_df.columns:
            if column not in preserve_columns:
                synthetic_df[column] = self.single_table_generator._generate_numeric_column(
                    synthetic_df[column], len(synthetic_df)
                ) if pd.api.types.is_numeric_dtype(synthetic_df[column]) else \
                self.single_table_generator._generate_categorical_column(
                    synthetic_df[column], len(synthetic_df)
                )

        return synthetic_df, new_mappings

    def _summarize_relationships(self, relationship_info: Dict) -> Dict:
        """Create human-readable relationship summary"""
        summary = {
            'total_relationships': len(relationship_info['relationships']),
            'tables_with_primary_keys': len([k for k, v in relationship_info['primary_keys'].items() if v]),
            'tables_with_foreign_keys': len(relationship_info['foreign_keys']),
            'generation_order': relationship_info['generation_order'],
            'relationship_details': []
        }

        for rel_key, rel_info in relationship_info['relationships'].items():
            summary['relationship_details'].append({
                'description': f"{rel_info['child_table']}.{rel_info['child_column']} references {rel_info['parent_table']}.{rel_info['parent_column']}",
                'confidence': round(rel_info['confidence'], 2)
            })

        return summary
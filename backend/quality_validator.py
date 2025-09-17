import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class QualityValidator:
    """Validate the quality of synthetic data by comparing with original data"""

    def __init__(self):
        self.statistical_tests = {
            'ks_test': self._kolmogorov_smirnov_test,
            'chi_square': self._chi_square_test,
            'correlation': self._correlation_comparison,
            'distribution': self._distribution_comparison
        }

    def compare_distributions(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive comparison between original and synthetic data"""

        logger.info("Starting quality validation of synthetic data")

        validation_results = {
            'overall_quality_score': 0.0,
            'column_comparisons': {},
            'statistical_tests': {},
            'data_utility_metrics': {},
            'privacy_utility_tradeoff': {},
            'recommendations': [],
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }

        # Basic data shape comparison
        validation_results['data_shape'] = {
            'original_shape': original_df.shape,
            'synthetic_shape': synthetic_df.shape,
            'shape_match': original_df.shape[1] == synthetic_df.shape[1]
        }

        # Column-by-column comparison
        common_columns = set(original_df.columns) & set(synthetic_df.columns)

        for column in common_columns:
            validation_results['column_comparisons'][column] = self._compare_column(
                original_df[column],
                synthetic_df[column],
                column
            )

        # Overall statistical tests
        validation_results['statistical_tests'] = self._run_statistical_tests(
            original_df,
            synthetic_df
        )

        # Data utility metrics
        validation_results['data_utility_metrics'] = self._calculate_utility_metrics(
            original_df,
            synthetic_df
        )

        # Calculate overall quality score
        validation_results['overall_quality_score'] = self._calculate_overall_score(
            validation_results
        )

        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(
            validation_results
        )

        logger.info(f"Quality validation completed. Overall score: {validation_results['overall_quality_score']}")

        return validation_results

    def _compare_column(self, original_series: pd.Series, synthetic_series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Compare a single column between original and synthetic data"""

        comparison = {
            'column_name': column_name,
            'data_type': str(original_series.dtype),
            'quality_score': 0.0,
            'metrics': {}
        }

        # Basic statistics
        comparison['metrics']['basic_stats'] = {
            'original_count': len(original_series.dropna()),
            'synthetic_count': len(synthetic_series.dropna()),
            'original_null_rate': original_series.isnull().mean(),
            'synthetic_null_rate': synthetic_series.isnull().mean()
        }

        if pd.api.types.is_numeric_dtype(original_series):
            comparison.update(self._compare_numeric_column(original_series, synthetic_series))
        elif pd.api.types.is_categorical_dtype(original_series) or original_series.dtype == 'object':
            comparison.update(self._compare_categorical_column(original_series, synthetic_series))
        elif pd.api.types.is_datetime64_any_dtype(original_series):
            comparison.update(self._compare_datetime_column(original_series, synthetic_series))

        return comparison

    def _compare_numeric_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Compare numeric columns"""

        orig_clean = original.dropna()
        synth_clean = synthetic.dropna()

        if len(orig_clean) == 0 or len(synth_clean) == 0:
            return {'quality_score': 0.0, 'metrics': {'error': 'No valid numeric data'}}

        comparison = {'metrics': {}}

        # Statistical measures
        stats_comparison = {
            'mean': {
                'original': float(orig_clean.mean()),
                'synthetic': float(synth_clean.mean()),
                'difference': float(abs(orig_clean.mean() - synth_clean.mean())),
                'relative_error': float(abs(orig_clean.mean() - synth_clean.mean()) / abs(orig_clean.mean()) if orig_clean.mean() != 0 else 0)
            },
            'std': {
                'original': float(orig_clean.std()),
                'synthetic': float(synth_clean.std()),
                'difference': float(abs(orig_clean.std() - synth_clean.std())),
                'relative_error': float(abs(orig_clean.std() - synth_clean.std()) / abs(orig_clean.std()) if orig_clean.std() != 0 else 0)
            },
            'median': {
                'original': float(orig_clean.median()),
                'synthetic': float(synth_clean.median()),
                'difference': float(abs(orig_clean.median() - synth_clean.median())),
                'relative_error': float(abs(orig_clean.median() - synth_clean.median()) / abs(orig_clean.median()) if orig_clean.median() != 0 else 0)
            },
            'min': {
                'original': float(orig_clean.min()),
                'synthetic': float(synth_clean.min())
            },
            'max': {
                'original': float(orig_clean.max()),
                'synthetic': float(synth_clean.max())
            }
        }

        comparison['metrics']['statistical_measures'] = stats_comparison

        # Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = ks_2samp(orig_clean, synth_clean)
            comparison['metrics']['ks_test'] = {
                'statistic': float(ks_statistic),
                'p_value': float(ks_p_value),
                'distributions_similar': ks_p_value > 0.05
            }
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            comparison['metrics']['ks_test'] = {'error': str(e)}

        # Distribution comparison
        comparison['metrics']['distribution'] = self._compare_distributions_detailed(orig_clean, synth_clean)

        # Calculate quality score for numeric column
        quality_score = 100.0

        # Penalize for large differences in statistical measures
        for measure in ['mean', 'std', 'median']:
            rel_error = stats_comparison[measure]['relative_error']
            if rel_error > 0.5:  # 50% difference
                quality_score -= 20
            elif rel_error > 0.2:  # 20% difference
                quality_score -= 10
            elif rel_error > 0.1:  # 10% difference
                quality_score -= 5

        # Bonus for similar distributions (KS test)
        if 'ks_test' in comparison['metrics'] and 'distributions_similar' in comparison['metrics']['ks_test']:
            if comparison['metrics']['ks_test']['distributions_similar']:
                quality_score += 10
            else:
                quality_score -= 15

        comparison['quality_score'] = max(0, min(100, quality_score))

        return comparison

    def _compare_categorical_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Compare categorical columns"""

        orig_clean = original.dropna()
        synth_clean = synthetic.dropna()

        comparison = {'metrics': {}}

        # Value counts comparison
        orig_counts = orig_clean.value_counts(normalize=True)
        synth_counts = synth_clean.value_counts(normalize=True)

        # Unique values
        orig_unique = set(orig_clean.unique())
        synth_unique = set(synth_clean.unique())

        comparison['metrics']['unique_values'] = {
            'original_count': len(orig_unique),
            'synthetic_count': len(synth_unique),
            'common_values': len(orig_unique & synth_unique),
            'original_only': list(orig_unique - synth_unique)[:10],  # Show first 10
            'synthetic_only': list(synth_unique - orig_unique)[:10],  # Show first 10
            'jaccard_similarity': len(orig_unique & synth_unique) / len(orig_unique | synth_unique) if orig_unique | synth_unique else 0
        }

        # Frequency distribution comparison
        comparison['metrics']['frequency_distribution'] = self._compare_frequency_distributions(
            orig_counts, synth_counts
        )

        # Chi-square test if applicable
        if len(orig_unique & synth_unique) > 1:
            try:
                chi2_result = self._chi_square_test_categorical(orig_clean, synth_clean)
                comparison['metrics']['chi_square_test'] = chi2_result
            except Exception as e:
                logger.warning(f"Chi-square test failed: {e}")
                comparison['metrics']['chi_square_test'] = {'error': str(e)}

        # Calculate quality score for categorical column
        quality_score = 100.0

        # Penalize for missing common values
        jaccard_sim = comparison['metrics']['unique_values']['jaccard_similarity']
        if jaccard_sim < 0.3:
            quality_score -= 30
        elif jaccard_sim < 0.5:
            quality_score -= 20
        elif jaccard_sim < 0.7:
            quality_score -= 10

        # Penalize for very different frequency distributions
        if 'frequency_distribution' in comparison['metrics']:
            freq_similarity = comparison['metrics']['frequency_distribution']['similarity_score']
            quality_score += (freq_similarity - 50) * 0.5  # Adjust based on frequency similarity

        comparison['quality_score'] = max(0, min(100, quality_score))

        return comparison

    def _compare_datetime_column(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Compare datetime columns"""

        orig_clean = pd.to_datetime(original.dropna())
        synth_clean = pd.to_datetime(synthetic.dropna())

        comparison = {'metrics': {}}

        if len(orig_clean) == 0 or len(synth_clean) == 0:
            return {'quality_score': 0.0, 'metrics': {'error': 'No valid datetime data'}}

        # Date range comparison
        comparison['metrics']['date_range'] = {
            'original_min': orig_clean.min().isoformat(),
            'original_max': orig_clean.max().isoformat(),
            'synthetic_min': synth_clean.min().isoformat(),
            'synthetic_max': synth_clean.max().isoformat(),
            'original_span_days': (orig_clean.max() - orig_clean.min()).days,
            'synthetic_span_days': (synth_clean.max() - synth_clean.min()).days
        }

        # Temporal patterns
        orig_year_counts = orig_clean.dt.year.value_counts(normalize=True)
        synth_year_counts = synth_clean.dt.year.value_counts(normalize=True)

        comparison['metrics']['temporal_patterns'] = {
            'year_distribution_similarity': self._calculate_distribution_similarity(orig_year_counts, synth_year_counts)
        }

        # Calculate quality score
        quality_score = 70.0  # Base score for datetime

        # Check if synthetic dates are within reasonable range
        orig_min, orig_max = orig_clean.min(), orig_clean.max()
        synth_min, synth_max = synth_clean.min(), synth_clean.max()

        if synth_min >= orig_min and synth_max <= orig_max:
            quality_score += 20  # Dates within original range
        elif synth_min < orig_min or synth_max > orig_max:
            quality_score -= 10  # Dates outside original range

        comparison['quality_score'] = max(0, min(100, quality_score))

        return comparison

    def _compare_distributions_detailed(self, orig: pd.Series, synth: pd.Series) -> Dict[str, Any]:
        """Detailed distribution comparison for numeric data"""

        # Create histograms
        bins = min(20, max(5, len(orig) // 10))  # Adaptive bin count

        try:
            orig_hist, bin_edges = np.histogram(orig, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth, bins=bin_edges, density=True)

            # Calculate histogram similarity (using chi-square distance)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            chi_square_distance = np.sum((orig_hist - synth_hist)**2 / (orig_hist + synth_hist + epsilon))

            # Normalize to 0-100 scale
            similarity_score = max(0, 100 - chi_square_distance * 10)

            return {
                'histogram_similarity': float(similarity_score),
                'chi_square_distance': float(chi_square_distance),
                'bins_used': int(bins)
            }

        except Exception as e:
            logger.warning(f"Distribution comparison failed: {e}")
            return {'error': str(e)}

    def _compare_frequency_distributions(self, orig_counts: pd.Series, synth_counts: pd.Series) -> Dict[str, Any]:
        """Compare frequency distributions of categorical data"""

        all_categories = set(orig_counts.index) | set(synth_counts.index)

        # Align the distributions
        orig_aligned = orig_counts.reindex(all_categories, fill_value=0)
        synth_aligned = synth_counts.reindex(all_categories, fill_value=0)

        # Calculate similarity metrics
        # Jensen-Shannon divergence
        js_divergence = self._jensen_shannon_divergence(orig_aligned.values, synth_aligned.values)

        # Total variation distance
        tv_distance = 0.5 * np.sum(np.abs(orig_aligned.values - synth_aligned.values))

        # Similarity score (0-100)
        similarity_score = 100 * (1 - js_divergence)

        return {
            'jensen_shannon_divergence': float(js_divergence),
            'total_variation_distance': float(tv_distance),
            'similarity_score': float(max(0, similarity_score)),
            'categories_compared': len(all_categories)
        }

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two probability distributions"""

        # Normalize to ensure they sum to 1
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        # Re-normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate JS divergence
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        js_divergence = 0.5 * kl_pm + 0.5 * kl_qm

        return float(js_divergence)

    def _calculate_distribution_similarity(self, dist1: pd.Series, dist2: pd.Series) -> float:
        """Calculate similarity between two distributions"""

        all_keys = set(dist1.index) | set(dist2.index)
        aligned1 = dist1.reindex(all_keys, fill_value=0)
        aligned2 = dist2.reindex(all_keys, fill_value=0)

        # Calculate cosine similarity
        dot_product = np.dot(aligned1.values, aligned2.values)
        norm1 = np.linalg.norm(aligned1.values)
        norm2 = np.linalg.norm(aligned2.values)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)
        return float(cosine_sim * 100)  # Convert to 0-100 scale

    def _chi_square_test_categorical(self, orig: pd.Series, synth: pd.Series) -> Dict[str, Any]:
        """Perform chi-square test for categorical data"""

        # Create contingency table
        orig_counts = orig.value_counts()
        synth_counts = synth.value_counts()

        all_categories = set(orig_counts.index) | set(synth_counts.index)

        orig_aligned = orig_counts.reindex(all_categories, fill_value=0)
        synth_aligned = synth_counts.reindex(all_categories, fill_value=0)

        # Create 2x2 contingency table
        contingency_table = np.array([orig_aligned.values, synth_aligned.values])

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            return {
                'chi_square_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'distributions_similar': p_value > 0.05
            }
        except Exception as e:
            return {'error': str(e)}

    def _run_statistical_tests(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Run overall statistical tests on the datasets"""

        tests_results = {}

        # Correlation matrix comparison for numeric columns
        numeric_columns = original_df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 1:
            orig_corr = original_df[numeric_columns].corr()
            synth_corr = synthetic_df[numeric_columns].corr()

            # Calculate correlation matrix similarity
            corr_diff = np.abs(orig_corr.values - synth_corr.values)
            mean_corr_diff = np.mean(corr_diff[~np.isnan(corr_diff)])

            tests_results['correlation_analysis'] = {
                'mean_correlation_difference': float(mean_corr_diff),
                'correlation_preservation_score': float(max(0, 100 - mean_corr_diff * 100)),
                'numeric_columns_analyzed': len(numeric_columns)
            }

        return tests_results

    def _calculate_utility_metrics(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data utility metrics"""

        utility_metrics = {
            'data_completeness': {
                'original_completeness': float((1 - original_df.isnull().mean().mean()) * 100),
                'synthetic_completeness': float((1 - synthetic_df.isnull().mean().mean()) * 100)
            },
            'data_diversity': {
                'original_unique_rows': len(original_df.drop_duplicates()),
                'synthetic_unique_rows': len(synthetic_df.drop_duplicates()),
                'diversity_preservation': 0.0
            },
            'statistical_fidelity': 0.0
        }

        # Calculate diversity preservation
        orig_diversity = utility_metrics['data_diversity']['original_unique_rows'] / len(original_df) if len(original_df) > 0 else 0
        synth_diversity = utility_metrics['data_diversity']['synthetic_unique_rows'] / len(synthetic_df) if len(synthetic_df) > 0 else 0

        utility_metrics['data_diversity']['diversity_preservation'] = float(min(synth_diversity / orig_diversity, 1.0) * 100 if orig_diversity > 0 else 0)

        return utility_metrics

    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""

        if not validation_results['column_comparisons']:
            return 0.0

        # Average column quality scores
        column_scores = [comp['quality_score'] for comp in validation_results['column_comparisons'].values()]
        avg_column_score = np.mean(column_scores) if column_scores else 0

        # Statistical tests bonus/penalty
        statistical_bonus = 0
        if 'correlation_analysis' in validation_results['statistical_tests']:
            corr_score = validation_results['statistical_tests']['correlation_analysis']['correlation_preservation_score']
            statistical_bonus = (corr_score - 50) * 0.2  # Small bonus/penalty

        # Data utility bonus
        utility_bonus = 0
        if 'data_diversity' in validation_results['data_utility_metrics']:
            diversity_score = validation_results['data_utility_metrics']['data_diversity']['diversity_preservation']
            utility_bonus = (diversity_score - 50) * 0.1

        overall_score = avg_column_score + statistical_bonus + utility_bonus

        return float(max(0, min(100, overall_score)))

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []
        overall_score = validation_results['overall_quality_score']

        if overall_score < 50:
            recommendations.append("Overall data quality is low. Consider using a different synthetic data generation method.")
        elif overall_score < 70:
            recommendations.append("Data quality is moderate. Fine-tuning the generation parameters may improve results.")
        else:
            recommendations.append("Good data quality achieved. The synthetic data preserves most statistical properties.")

        # Column-specific recommendations
        low_quality_columns = [
            col for col, comp in validation_results['column_comparisons'].items()
            if comp['quality_score'] < 60
        ]

        if low_quality_columns:
            recommendations.append(f"The following columns have low quality and may need attention: {', '.join(low_quality_columns[:5])}")

        # Statistical tests recommendations
        if 'correlation_analysis' in validation_results['statistical_tests']:
            corr_score = validation_results['statistical_tests']['correlation_analysis']['correlation_preservation_score']
            if corr_score < 70:
                recommendations.append("Correlation structure is not well preserved. Consider using more advanced generation methods like CTGAN.")

        # Privacy-utility tradeoff
        if overall_score > 80:
            recommendations.append("High utility achieved. Ensure adequate privacy protection is still maintained.")

        return recommendations

    def _kolmogorov_smirnov_test(self, orig: pd.Series, synth: pd.Series) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test implementation"""
        try:
            statistic, p_value = ks_2samp(orig.dropna(), synth.dropna())
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'null_hypothesis_accepted': p_value > 0.05
            }
        except Exception as e:
            return {'error': str(e)}

    def _chi_square_test(self, orig: pd.Series, synth: pd.Series) -> Dict[str, Any]:
        """Chi-square test implementation"""
        return self._chi_square_test_categorical(orig, synth)

    def _correlation_comparison(self, orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare correlation matrices"""
        numeric_cols = orig_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}

        orig_corr = orig_df[numeric_cols].corr()
        synth_corr = synth_df[numeric_cols].corr()

        correlation_diff = np.abs(orig_corr.values - synth_corr.values)
        mean_diff = np.mean(correlation_diff[~np.isnan(correlation_diff)])

        return {
            'mean_correlation_difference': float(mean_diff),
            'preservation_score': float(max(0, 100 - mean_diff * 100))
        }

    def _distribution_comparison(self, orig: pd.Series, synth: pd.Series) -> Dict[str, Any]:
        """Compare distributions"""
        if pd.api.types.is_numeric_dtype(orig):
            return self._compare_distributions_detailed(orig.dropna(), synth.dropna())
        else:
            orig_counts = orig.value_counts(normalize=True)
            synth_counts = synth.value_counts(normalize=True)
            return self._compare_frequency_distributions(orig_counts, synth_counts)
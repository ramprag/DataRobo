import React, { useState } from 'react';
import './QualityReport.css';

const QualityReport = ({ qualityMetrics, privacyConfig }) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (!qualityMetrics) {
    return (
      <div className="quality-report">
        <div className="report-placeholder">
          <p>Quality report will be available after synthetic data generation</p>
        </div>
      </div>
    );
  }

  const getScoreColor = (score) => {
    if (score >= 80) return 'score-excellent';
    if (score >= 60) return 'score-good';
    if (score >= 40) return 'score-fair';
    return 'score-poor';
  };

  const getScoreLabel = (score) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Improvement';
  };

  const overallScore = qualityMetrics.overall_quality_score || 0;

  return (
    <div className="quality-report">
      <div className="report-header">
        <h3>📊 Data Quality Report</h3>
        <div className={`overall-score ${getScoreColor(overallScore)}`}>
          <div className="score-circle">
            <span className="score-value">{Math.round(overallScore)}</span>
            <span className="score-unit">%</span>
          </div>
          <div className="score-label">{getScoreLabel(overallScore)}</div>
        </div>
      </div>

      <div className="report-tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📋 Overview
        </button>
        <button
          className={`tab ${activeTab === 'privacy' ? 'active' : ''}`}
          onClick={() => setActiveTab('privacy')}
        >
          🔒 Privacy
        </button>
      </div>

      <div className="report-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            <div className="metrics-grid">
              <div className="metric-card">
                <h4>📏 Data Dimensions</h4>
                <div className="metric-value">
                  {qualityMetrics.data_shape?.original_shape?.[0]?.toLocaleString() || '—'} rows × {' '}
                  {qualityMetrics.data_shape?.original_shape?.[1] || '—'} columns
                </div>
                <div className="metric-comparison">
                  Synthetic: {qualityMetrics.data_shape?.synthetic_shape?.[0]?.toLocaleString() || '—'} rows × {' '}
                  {qualityMetrics.data_shape?.synthetic_shape?.[1] || '—'} columns
                </div>
              </div>

              {qualityMetrics.data_utility_metrics?.data_completeness && (
                <div className="metric-card">
                  <h4>✅ Data Completeness</h4>
                  <div className="metric-value">
                    {Math.round(qualityMetrics.data_utility_metrics.data_completeness.synthetic_completeness)}%
                  </div>
                  <div className="metric-comparison">
                    Original: {Math.round(qualityMetrics.data_utility_metrics.data_completeness.original_completeness)}%
                  </div>
                </div>
              )}
            </div>

            {qualityMetrics.recommendations && (
              <div className="insights-section">
                <h4>💡 Recommendations</h4>
                <div className="insights">
                  {qualityMetrics.recommendations.map((rec, i) => (
                    <div key={i} className="insight insight-positive">
                      <span className="insight-icon">✅</span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'privacy' && (
          <div className="privacy-tab">
            <h4>🔒 Privacy Protection Summary</h4>
            {privacyConfig && (
              <div className="privacy-score-card">
                <h5>Privacy Configuration Applied</h5>
                <div className="privacy-settings">
                  <div className={`privacy-setting ${privacyConfig.mask_emails ? 'enabled' : 'disabled'}`}>
                    <span className="setting-icon">📧</span>
                    <span className="setting-name">Email Masking</span>
                    <span className="setting-status">{privacyConfig.mask_emails ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className={`privacy-setting ${privacyConfig.mask_names ? 'enabled' : 'disabled'}`}>
                    <span className="setting-icon">👤</span>
                    <span className="setting-name">Name Masking</span>
                    <span className="setting-status">{privacyConfig.mask_names ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className={`privacy-setting ${privacyConfig.mask_phone_numbers ? 'enabled' : 'disabled'}`}>
                    <span className="setting-icon">📱</span>
                    <span className="setting-name">Phone Masking</span>
                    <span className="setting-status">{privacyConfig.mask_phone_numbers ? 'Enabled' : 'Disabled'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default QualityReport;
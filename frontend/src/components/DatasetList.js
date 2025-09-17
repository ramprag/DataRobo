import React from 'react';
import './DatasetList.css';

const DatasetList = ({ datasets, selectedDataset, onDatasetSelect, onDatasetDelete, loading }) => {

  const getStatusIcon = (status) => {
    switch (status) {
      case 'uploaded':
        return '📤';
      case 'processing':
        return '⚙️';
      case 'completed':
        return '✅';
      case 'failed':
        return '❌';
      default:
        return '📄';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'uploaded':
        return 'status-uploaded';
      case 'processing':
        return 'status-processing';
      case 'completed':
        return 'status-completed';
      case 'failed':
        return 'status-failed';
      default:
        return 'status-unknown';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (loading && datasets.length === 0) {
    return (
      <div className="dataset-list loading">
        <div className="loading-spinner">
          <span className="spinner"></span>
          <p>Loading datasets...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dataset-list">
      <div className="list-header">
        <h3>📊 Your Datasets</h3>
        <span className="dataset-count">{datasets.length} dataset{datasets.length !== 1 ? 's' : ''}</span>
      </div>

      {datasets.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📂</div>
          <p>No datasets yet</p>
          <small>Upload your first dataset to get started</small>
        </div>
      ) : (
        <div className="dataset-items">
          {datasets.map((dataset) => (
            <div
              key={dataset.id}
              className={`dataset-item ${selectedDataset?.id === dataset.id ? 'selected' : ''}`}
              onClick={() => onDatasetSelect(dataset)}
            >
              <div className="dataset-header">
                <div className="dataset-icon">
                  {getStatusIcon(dataset.status)}
                </div>
                <div className="dataset-info">
                  <div className="dataset-name" title={dataset.filename}>
                    {dataset.filename}
                  </div>
                  <div className={`dataset-status ${getStatusColor(dataset.status)}`}>
                    {dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}
                    {dataset.status === 'processing' && (
                      <span className="processing-spinner"></span>
                    )}
                  </div>
                </div>
                <div className="dataset-actions">
                  <button
                    className="btn btn-icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDatasetDelete(dataset.id);
                    }}
                    title="Delete dataset"
                  >
                    🗑️
                  </button>
                </div>
              </div>

              <div className="dataset-meta">
                <div className="meta-item">
                  <span className="meta-label">📊</span>
                  <span className="meta-value">
                    {dataset.row_count?.toLocaleString() || '—'} rows
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">📋</span>
                  <span className="meta-value">
                    {dataset.column_count || '—'} columns
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">💾</span>
                  <span className="meta-value">
                    {formatFileSize(dataset.file_size)}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">🕒</span>
                  <span className="meta-value">
                    {formatDate(dataset.created_at)}
                  </span>
                </div>
              </div>

              {dataset.status === 'failed' && dataset.error_message && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  <span className="error-text">{dataset.error_message}</span>
                </div>
              )}

              {dataset.status === 'completed' && dataset.quality_metrics && (
                <div className="quality-preview">
                  <span className="quality-label">Quality Score:</span>
                  <span className="quality-score">
                    {Math.round(dataset.quality_metrics.overall_quality_score || 0)}%
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="status-legend">
        <h4>Status Legend:</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-icon">📤</span>
            <span className="legend-text">Uploaded</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">⚙️</span>
            <span className="legend-text">Processing</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">✅</span>
            <span className="legend-text">Completed</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">❌</span>
            <span className="legend-text">Failed</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetList;
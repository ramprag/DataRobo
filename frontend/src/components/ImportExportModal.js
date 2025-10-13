// frontend/src/components/ImportExportModal.js

import React, { useState } from 'react';
import './ImportExportModal.css';

const ImportExportModal = ({ show, mode, onClose, onSubmit, selectedDataset }) => {
  const [selectedSource, setSelectedSource] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  const dataSources = [
    {
      id: 's3',
      name: 'Amazon S3',
      icon: '☁️',
      color: '#FF9900',
      description: 'Import from or export to S3 buckets with automatic credential management and encryption',
      comingSoon: true,
      features: ['Automatic encryption', 'IAM role support', 'Bucket versioning']
    },
    {
      id: 'azure',
      name: 'Azure Blob Storage',
      icon: '☁️',
      color: '#0078D4',
      description: 'Connect to Azure Blob Storage with support for all container types and access tiers',
      comingSoon: true,
      features: ['SAS token support', 'Hot/Cool/Archive tiers', 'Encryption at rest']
    },
    {
      id: 'database',
      name: 'Database Connection',
      icon: '🗄️',
      color: '#9333EA',
      description: 'Direct connection to PostgreSQL, MySQL, SQL Server, Oracle, and more',
      comingSoon: true,
      features: ['SSL connections', 'Query optimization', 'Batch processing']
    },
    {
      id: 'local',
      name: 'Local File System',
      icon: '💾',
      color: '#10B981',
      description: 'Upload CSV, JSON, Excel, or Parquet files directly from your computer',
      comingSoon: false,
      features: ['Drag & drop support', 'Multiple formats', 'Up to 100MB']
    },
  ];

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleSubmit = () => {
    if (mode === 'import' && selectedSource === 'local' && selectedFile) {
      onSubmit(selectedSource, selectedFile);
    } else if (mode === 'export' && selectedSource) {
      onSubmit(selectedSource);
    } else if (selectedSource) {
      onSubmit(selectedSource, null);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (!show) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>

        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title">
            <span className="modal-icon">{mode === 'import' ? '📥' : '📤'}</span>
            <h2>{mode === 'import' ? 'Import Data' : 'Export Data'}</h2>
          </div>
          <button onClick={onClose} className="modal-close">×</button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">

          {/* Info Banner */}
          <div className="info-banner">
            <span className="info-icon">💡</span>
            <p>
              {mode === 'import'
                ? 'Select a data source to import your existing data. GSTAN will analyze the schema and help you generate synthetic data based on it.'
                : `Choose where to export "${selectedDataset?.filename || 'your dataset'}". All exports maintain data integrity and format consistency.`
              }
            </p>
          </div>

          {/* Data Sources Grid */}
          <div className="data-sources-grid">
            {dataSources.map((source) => (
              <button
                key={source.id}
                onClick={() => !source.comingSoon && setSelectedSource(source.id)}
                disabled={source.comingSoon}
                className={`data-source-card ${selectedSource === source.id ? 'selected' : ''} ${source.comingSoon ? 'disabled' : ''}`}
              >
                {source.comingSoon && (
                  <div className="coming-soon-badge">Coming Soon</div>
                )}

                <div className="source-header">
                  <span className="source-icon" style={{ color: source.color }}>
                    {source.icon}
                  </span>
                  <h3>{source.name}</h3>
                </div>

                <p className="source-description">{source.description}</p>

                <div className="source-features">
                  {source.features.map((feature, idx) => (
                    <span key={idx} className="feature-tag">
                      ✓ {feature}
                    </span>
                  ))}
                </div>

                {selectedSource === source.id && !source.comingSoon && (
                  <div className="selected-indicator">
                    <span>✓</span> Selected
                  </div>
                )}
              </button>
            ))}
          </div>

          {/* File Upload Section (for local import) */}
          {mode === 'import' && selectedSource === 'local' && (
            <div className="file-upload-section">
              <h4>📁 Select File to Upload</h4>

              <div className="file-input-wrapper">
                <input
                  type="file"
                  id="modal-file-input"
                  accept=".csv,.xlsx,.xls,.json,.parquet,.zip"
                  onChange={handleFileSelect}
                  className="file-input"
                />
                <label htmlFor="modal-file-input" className="file-input-label">
                  <span className="upload-icon">📤</span>
                  <span>Click to browse files</span>
                </label>
              </div>

              {selectedFile && (
                <div className="selected-file-info">
                  <div className="file-details">
                    <span className="file-icon">📄</span>
                    <div>
                      <div className="file-name">{selectedFile.name}</div>
                      <div className="file-size">{formatFileSize(selectedFile.size)}</div>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="remove-file-btn"
                  >
                    ×
                  </button>
                </div>
              )}

              <div className="file-format-note">
                <strong>Supported formats:</strong> CSV, Excel (.xlsx, .xls), JSON, Parquet, ZIP (multi-table)
              </div>
            </div>
          )}

          {/* Connection Details (for future sources) */}
          {selectedSource && selectedSource !== 'local' && (
            <div className="connection-details">
              <div className="future-feature-notice">
                <span className="notice-icon">🚀</span>
                <div>
                  <h4>Feature Under Development</h4>
                  <p>
                    {dataSources.find(s => s.id === selectedSource)?.name} integration is coming soon!
                    Our team is working on implementing secure connections with enterprise-grade features.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button onClick={onClose} className="btn btn-secondary">
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={
              !selectedSource ||
              (mode === 'import' && selectedSource === 'local' && !selectedFile)
            }
            className="btn btn-primary"
            style={{
              background: selectedSource && (mode === 'export' || selectedFile)
                ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                : undefined
            }}
          >
            {mode === 'import' ? '📥 Import' : '📤 Export'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImportExportModal;
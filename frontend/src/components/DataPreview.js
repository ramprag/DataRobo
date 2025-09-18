import React, { useState, useEffect } from 'react';
import './DataPreview.css';

const DataPreview = ({ datasetId, title, synthetic = false }) => {
  const [previewData, setPreviewData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (datasetId) {
      fetchPreview();
    }
  }, [datasetId, synthetic]);

  const fetchPreview = async () => {
    try {
      setLoading(true);
      setError(null);

      // Use relative URL that will work with the proxy
      const apiUrl = process.env.REACT_APP_API_URL || '';
      const url = `${apiUrl}/api/datasets/${datasetId}/preview?synthetic=${synthetic}`;

      console.log('Fetching preview from:', url);

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Preview fetch failed:', response.status, errorText);
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to load preview'}`);
      }

      const data = await response.json();
      console.log('Preview data received:', data);
      setPreviewData(data);
    } catch (err) {
      console.error('Failed to fetch preview:', err);
      setError(err.message || 'Failed to load preview');
    } finally {
      setLoading(false);
    }
  };

  const renderTableData = (data) => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return (
        <tr>
          <td colSpan={previewData?.columns?.length || 1} style={{ textAlign: 'center', padding: '2rem' }}>
            No data available
          </td>
        </tr>
      );
    }

    return data.map((row, index) => (
      <tr key={index}>
        {previewData.columns?.map((column, colIndex) => (
          <td key={colIndex} title={String(row[column] || '—')}>
            {String(row[column] || '—')}
          </td>
        ))}
      </tr>
    ));
  };

  if (loading) {
    return (
      <div className="data-preview loading">
        <h4>{title}</h4>
        <div className="loading-spinner">
          <span className="spinner"></span>
          <p>Loading preview...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="data-preview error">
        <h4>{title}</h4>
        <div className="error-message">
          <span className="error-icon">❌</span>
          <p>{error}</p>
          <button
            onClick={fetchPreview}
            className="btn btn-text"
            style={{ marginTop: '0.5rem' }}
          >
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  if (!previewData) {
    return (
      <div className="data-preview empty">
        <h4>{title}</h4>
        <p>No preview available</p>
      </div>
    );
  }

  return (
    <div className="data-preview">
      <div className="preview-header">
        <h4>{title}</h4>
        <div className="preview-stats">
          <span>📊 {previewData.total_rows?.toLocaleString() || 'Unknown'} total rows</span>
          <span>👁️ Showing {previewData.preview_rows || previewData.data?.length || 0} rows</span>
        </div>
      </div>

      <div className="preview-table-container">
        <table className="preview-table">
          <thead>
            <tr>
              {previewData.columns?.map((column, index) => (
                <th key={index}>{column}</th>
              )) || (
                <th>No columns available</th>
              )}
            </tr>
          </thead>
          <tbody>
            {renderTableData(previewData.data)}
          </tbody>
        </table>
      </div>

      {previewData.data && previewData.data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '1rem', color: '#6b7280', fontStyle: 'italic' }}>
          No data rows available in this dataset
        </div>
      )}
    </div>
  );
};

export default DataPreview;
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

      const response = await fetch(
        `/api/datasets/${datasetId}/preview?synthetic=${synthetic}`
      );

      if (response.ok) {
        const data = await response.json();
        setPreviewData(data);
      } else {
        throw new Error('Failed to load preview');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
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
          <span>📊 {previewData.total_rows?.toLocaleString()} total rows</span>
          <span>👁️ Showing {previewData.preview_rows} rows</span>
        </div>
      </div>

      <div className="preview-table-container">
        <table className="preview-table">
          <thead>
            <tr>
              {previewData.columns?.map((column, index) => (
                <th key={index}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewData.data?.map((row, index) => (
              <tr key={index}>
                {previewData.columns?.map((column, colIndex) => (
                  <td key={colIndex} title={row[column]}>
                    {String(row[column] || '—')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataPreview;
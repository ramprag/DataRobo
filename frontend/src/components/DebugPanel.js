import React from 'react';

const DebugPanel = ({ selectedDataset, currentStep, loading, error, success }) => {
  // Only show in development
  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      backgroundColor: '#1f2937',
      color: '#f9fafb',
      padding: '12px',
      borderRadius: '8px',
      fontSize: '12px',
      maxWidth: '300px',
      zIndex: 1000,
      fontFamily: 'monospace'
    }}>
      <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
        🐛 Debug Panel
      </div>

      <div>
        <strong>Current Step:</strong> {currentStep}
      </div>

      <div>
        <strong>Loading:</strong> {loading ? 'Yes' : 'No'}
      </div>

      <div>
        <strong>Selected Dataset:</strong> {selectedDataset?.id || 'None'}
      </div>

      {selectedDataset && (
        <div>
          <strong>Dataset Status:</strong> {selectedDataset.status}
        </div>
      )}

      {error && (
        <div style={{ color: '#ef4444' }}>
          <strong>Error:</strong> {error.substring(0, 100)}...
        </div>
      )}

      {success && (
        <div style={{ color: '#10b981' }}>
          <strong>Success:</strong> {success.substring(0, 100)}...
        </div>
      )}

      <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
        Last Updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default DebugPanel;
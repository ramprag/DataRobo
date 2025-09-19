import React, { useState, useEffect } from 'react';
import './GenerationProgress.css';

const GenerationProgress = ({ dataset, onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Initializing...');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!dataset) return;

    console.log('GenerationProgress - Dataset status:', dataset.status);

    // Handle different dataset statuses
    if (dataset.status === 'completed') {
      console.log('Dataset is completed, setting progress to 100%');
      setProgress(100);
      setStatus('Generation completed successfully! 🎉');
      setError(null);

      // Call onComplete after a short delay
      setTimeout(() => {
        if (onComplete) {
          console.log('Calling onComplete callback');
          onComplete();
        }
      }, 1500);

    } else if (dataset.status === 'processing') {
      setStatus('Processing your data...');
      setError(null);

      // Start realistic progress simulation
      let currentProgress = progress;

      const progressInterval = setInterval(() => {
        setProgress(prev => {
          // Don't go beyond 95% while processing to avoid getting stuck
          const newProgress = Math.min(prev + (Math.random() * 8 + 2), 95);
          currentProgress = newProgress;

          // Update status based on progress
          if (newProgress < 25) {
            setStatus('📊 Analyzing data structure...');
          } else if (newProgress < 50) {
            setStatus('🔒 Applying privacy masks...');
          } else if (newProgress < 80) {
            setStatus('🎯 Generating synthetic data...');
          } else {
            setStatus('📋 Validating quality...');
          }

          return newProgress;
        });
      }, 1000); // Update every second

      // Clear interval after 30 seconds to prevent infinite running
      setTimeout(() => {
        clearInterval(progressInterval);
      }, 30000);

      return () => clearInterval(progressInterval);

    } else if (dataset.status === 'failed') {
      setStatus('Generation failed ❌');
      setProgress(0);
      setError(dataset.error_message || 'Unknown error occurred');

    } else if (dataset.status === 'uploaded') {
      setStatus('Ready to generate');
      setProgress(0);
      setError(null);
    }
  }, [dataset?.status, onComplete]);

  // Force completion detection if dataset status changes to completed
  useEffect(() => {
    console.log('Dataset status effect - Status:', dataset?.status, 'Current progress:', progress);

    if (dataset?.status === 'completed' && progress < 100) {
      console.log('Force completing - dataset is completed but progress < 100');
      setProgress(100);
      setStatus('Generation completed successfully! 🎉');
      setError(null);
    }
  }, [dataset?.status]);

  const getProgressColor = () => {
    if (error) return '#ef4444';
    if (progress >= 100) return '#10b981';
    return '#3b82f6';
  };

  return (
    <div className="generation-progress">
      <div className="progress-header">
        <h4>🔄 Processing Your Data</h4>
        <span className="progress-percentage" style={{ color: getProgressColor() }}>
          {Math.round(progress)}%
        </span>
      </div>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${progress}%`,
            backgroundColor: getProgressColor()
          }}
        ></div>
      </div>

      <div className="progress-status">
        <span className="status-text" style={{ color: error ? '#ef4444' : '#4b5563' }}>
          {status}
        </span>
        {dataset?.status === 'processing' && progress < 100 && !error && (
          <span className="processing-spinner"></span>
        )}
      </div>

      {error && (
        <div className="error-section" style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '6px',
          color: '#991b1b'
        }}>
          <strong>Error Details:</strong>
          <p>{error}</p>
        </div>
      )}

      <div className="progress-steps">
        <div className={`progress-step ${progress > 0 ? (progress > 25 ? 'completed' : 'active') : ''}`}>
          📊 Analyzing data structure
        </div>
        <div className={`progress-step ${progress > 25 ? (progress > 50 ? 'completed' : 'active') : ''}`}>
          🔒 Applying privacy masks
        </div>
        <div className={`progress-step ${progress > 50 ? (progress > 80 ? 'completed' : 'active') : ''}`}>
          🎯 Generating synthetic data
        </div>
        <div className={`progress-step ${progress >= 100 ? 'completed' : progress > 80 ? 'active' : ''}`}>
          📋 Validating quality
        </div>
      </div>

      <div className="estimated-time">
        {dataset?.status === 'processing' && progress < 100 && !error && (
          <p>⏱️ Processing in progress...</p>
        )}
        {dataset?.status === 'completed' && (
          <p>✅ Process completed successfully!</p>
        )}
        {error && (
          <p>❌ Process failed - please try again</p>
        )}
      </div>

      {/* Debug info - remove in production */}
      <div style={{
        marginTop: '1rem',
        padding: '0.5rem',
        backgroundColor: '#f3f4f6',
        borderRadius: '4px',
        fontSize: '0.75rem',
        color: '#6b7280'
      }}>
        <strong>Debug:</strong> Status: {dataset?.status}, Progress: {progress}%, Error: {error ? 'Yes' : 'No'}
      </div>
    </div>
  );
};

export default GenerationProgress;
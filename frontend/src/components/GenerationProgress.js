import React, { useState, useEffect } from 'react';
import './GenerationProgress.css';

const GenerationProgress = ({ dataset, onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    if (dataset?.status === 'completed') {
      setProgress(100);
      setStatus('Generation completed!');
      if (onComplete) onComplete();
    } else if (dataset?.status === 'processing') {
      setStatus('Generating synthetic data...');
      // Simulate progress for visual feedback
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 1000);
      return () => clearInterval(interval);
    } else if (dataset?.status === 'failed') {
      setStatus('Generation failed');
      setProgress(0);
    }
  }, [dataset, onComplete]);

  return (
    <div className="generation-progress">
      <div className="progress-header">
        <h4>🔄 Processing Your Data</h4>
        <span className="progress-percentage">{Math.round(progress)}%</span>
      </div>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${progress}%` }}
        ></div>
      </div>

      <div className="progress-status">
        <span className="status-text">{status}</span>
        {dataset?.status === 'processing' && (
          <span className="processing-spinner"></span>
        )}
      </div>

      <div className="progress-steps">
        <div className={`progress-step ${progress > 20 ? 'completed' : 'active'}`}>
          📊 Analyzing data structure
        </div>
        <div className={`progress-step ${progress > 40 ? 'completed' : progress > 20 ? 'active' : ''}`}>
          🔒 Applying privacy masks
        </div>
        <div className={`progress-step ${progress > 70 ? 'completed' : progress > 40 ? 'active' : ''}`}>
          🎯 Generating synthetic data
        </div>
        <div className={`progress-step ${progress > 90 ? 'completed' : progress > 70 ? 'active' : ''}`}>
          📋 Validating quality
        </div>
      </div>

      <div className="estimated-time">
        {dataset?.status === 'processing' && (
          <p>⏱️ Estimated time: 2-5 minutes</p>
        )}
      </div>
    </div>
  );
};

export default GenerationProgress;
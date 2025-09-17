import React, { useState, useEffect } from 'react';
import './GenerationProgress.css';

const GenerationProgress = ({ dataset, onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    if (!dataset) return;

    console.log('GenerationProgress dataset status:', dataset.status); // Debug log

    if (dataset.status === 'completed') {
      setProgress(100);
      setStatus('Generation completed successfully! 🎉');
      // Call onComplete after a short delay to show 100%
      setTimeout(() => {
        if (onComplete) onComplete();
      }, 1000);
    } else if (dataset.status === 'processing') {
      setStatus('Generating synthetic data...');

      // Realistic progress simulation that doesn't get stuck
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          // Don't exceed 95% during processing to avoid getting stuck at 100%
          if (prev >= 95) return 95;

          // Simulate realistic progress steps
          let increment = 0;
          if (prev < 20) increment = Math.random() * 15; // Fast initial progress
          else if (prev < 60) increment = Math.random() * 10; // Steady progress
          else increment = Math.random() * 5; // Slower final steps

          return Math.min(prev + increment, 95);
        });

        // Update status messages based on progress
        setStatus(prevStatus => {
          if (progress < 25) return '📊 Analyzing data structure...';
          else if (progress < 50) return '🔒 Applying privacy masks...';
          else if (progress < 80) return '🎯 Generating synthetic data...';
          else return '📋 Validating quality...';
        });
      }, 1500);

      return () => clearInterval(progressInterval);
    } else if (dataset.status === 'failed') {
      setStatus('Generation failed ❌');
      setProgress(0);
    } else if (dataset.status === 'uploaded') {
      setStatus('Ready to generate');
      setProgress(0);
    }
  }, [dataset, onComplete]);

  // Force completion when dataset status changes to completed
  useEffect(() => {
    if (dataset?.status === 'completed' && progress < 100) {
      setProgress(100);
      setStatus('Generation completed successfully! 🎉');
    }
  }, [dataset?.status, progress]);

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
        {dataset?.status === 'processing' && progress < 100 && (
          <span className="processing-spinner"></span>
        )}
      </div>

      <div className="progress-steps">
        <div className={`progress-step ${progress > 20 ? 'completed' : progress > 0 ? 'active' : ''}`}>
          📊 Analyzing data structure
        </div>
        <div className={`progress-step ${progress > 40 ? 'completed' : progress > 20 ? 'active' : ''}`}>
          🔒 Applying privacy masks
        </div>
        <div className={`progress-step ${progress > 70 ? 'completed' : progress > 40 ? 'active' : ''}`}>
          🎯 Generating synthetic data
        </div>
        <div className={`progress-step ${progress >= 100 ? 'completed' : progress > 70 ? 'active' : ''}`}>
          📋 Validating quality
        </div>
      </div>

      <div className="estimated-time">
        {dataset?.status === 'processing' && progress < 100 && (
          <p>⏱️ Estimated time remaining: {Math.max(1, Math.round((100 - progress) / 20))} minute(s)</p>
        )}
        {dataset?.status === 'completed' && (
          <p>✅ Process completed successfully!</p>
        )}
      </div>
    </div>
  );
};

export default GenerationProgress;
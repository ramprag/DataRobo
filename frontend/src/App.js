import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Components
import FileUpload from './components/FileUpload';
import DatasetList from './components/DatasetList';
import PrivacyConfig from './components/PrivacyConfig';
import GenerationProgress from './components/GenerationProgress';
import QualityReport from './components/QualityReport';
import DataPreview from './components/DataPreview';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [currentStep, setCurrentStep] = useState('upload'); // upload, configure, generate, review
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Fetch datasets on component mount
  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/datasets');
      setDatasets(response.data);
    } catch (err) {
      setError('Failed to fetch datasets: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/datasets/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const newDataset = response.data;
      setDatasets(prev => [newDataset, ...prev]);
      setSelectedDataset(newDataset);
      setCurrentStep('configure');
      setSuccess('Dataset uploaded successfully!');

    } catch (err) {
      setError('Upload failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (dataset) => {
    setSelectedDataset(dataset);

    if (dataset.status === 'uploaded') {
      setCurrentStep('configure');
    } else if (dataset.status === 'processing') {
      setCurrentStep('generate');
    } else if (dataset.status === 'completed') {
      setCurrentStep('review');
    } else if (dataset.status === 'failed') {
      setCurrentStep('configure');
      setError(`Previous generation failed: ${dataset.error_message}`);
    }
  };

  const handlePrivacyConfig = async (privacyConfig, numRows = null) => {
    if (!selectedDataset) return;

    try {
      setLoading(true);
      setError(null);
      setCurrentStep('generate');

      const requestData = {
        dataset_id: selectedDataset.id,
        privacy_config: privacyConfig,
        num_rows: numRows
      };

      await api.post(`/api/datasets/${selectedDataset.id}/generate-synthetic`, requestData);

      // Start polling for progress
      pollGenerationStatus();

    } catch (err) {
      setError('Failed to start generation: ' + (err.response?.data?.detail || err.message));
      setCurrentStep('configure');
    } finally {
      setLoading(false);
    }
  };

// Replace the existing pollGenerationStatus function with this corrected version
const pollGenerationStatus = () => {
  const pollInterval = setInterval(async () => {
    if (!selectedDataset) {
      clearInterval(pollInterval);
      return;
    }

    try {
      const response = await api.get(`/api/datasets/${selectedDataset.id}`);
      const updatedDataset = response.data;

      console.log('Polling status:', updatedDataset.status); // Debug log

      setSelectedDataset(updatedDataset);

      // Update dataset in list
      setDatasets(prev =>
        prev.map(d => d.id === updatedDataset.id ? updatedDataset : d)
      );

      if (updatedDataset.status === 'completed') {
        clearInterval(pollInterval);
        setCurrentStep('review');
        setSuccess('Synthetic data generated successfully!');
        setLoading(false);
      } else if (updatedDataset.status === 'failed') {
        clearInterval(pollInterval);
        setCurrentStep('configure');
        setError(`Generation failed: ${updatedDataset.error_message}`);
        setLoading(false);
      }

    } catch (err) {
      console.error('Error polling status:', err);
      clearInterval(pollInterval);
      setError('Error checking generation status');
      setLoading(false);
    }
  }, 2000); // Poll every 2 seconds

  // Cleanup after 5 minutes to prevent infinite polling
  setTimeout(() => {
    clearInterval(pollInterval);
    if (selectedDataset?.status === 'processing') {
      setError('Generation is taking longer than expected. Please refresh and try again.');
      setLoading(false);
    }
  }, 300000); // 5 minutes timeout

  return pollInterval;
};

  const handleDownload = async () => {
    if (!selectedDataset) return;

    try {
      const response = await api.get(`/api/datasets/${selectedDataset.id}/download`, {
        responseType: 'blob',
      });

      // Create download link
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedDataset.filename}_synthetic.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSuccess('Synthetic data downloaded successfully!');

    } catch (err) {
      setError('Download failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDeleteDataset = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;

    try {
      await api.delete(`/api/datasets/${datasetId}`);
      setDatasets(prev => prev.filter(d => d.id !== datasetId));

      if (selectedDataset?.id === datasetId) {
        setSelectedDataset(null);
        setCurrentStep('upload');
      }

      setSuccess('Dataset deleted successfully!');

    } catch (err) {
      setError('Delete failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="container">
          <h1>🔒 Synthetic Data Generator</h1>
          <p>Privacy-safe synthetic data generation for your datasets</p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">

          {/* Status Messages */}
          {error && (
            <div className="alert alert-error">
              <span>{error}</span>
              <button onClick={clearMessages} className="alert-close">×</button>
            </div>
          )}

          {success && (
            <div className="alert alert-success">
              <span>{success}</span>
              <button onClick={clearMessages} className="alert-close">×</button>
            </div>
          )}

          {/* Progress Indicator */}
          <div className="progress-steps">
            <div className={`step ${currentStep === 'upload' ? 'active' : currentStep !== 'upload' ? 'completed' : ''}`}>
              <span className="step-number">1</span>
              <span className="step-label">Upload Dataset</span>
            </div>
            <div className={`step ${currentStep === 'configure' ? 'active' : currentStep === 'generate' || currentStep === 'review' ? 'completed' : ''}`}>
              <span className="step-number">2</span>
              <span className="step-label">Configure Privacy</span>
            </div>
            <div className={`step ${currentStep === 'generate' ? 'active' : currentStep === 'review' ? 'completed' : ''}`}>
              <span className="step-number">3</span>
              <span className="step-label">Generate Data</span>
            </div>
            <div className={`step ${currentStep === 'review' ? 'active' : ''}`}>
              <span className="step-number">4</span>
              <span className="step-label">Review & Download</span>
            </div>
          </div>

          <div className="app-content">
            {/* Sidebar - Dataset List */}
            <aside className="sidebar">
              <DatasetList
                datasets={datasets}
                selectedDataset={selectedDataset}
                onDatasetSelect={handleDatasetSelect}
                onDatasetDelete={handleDeleteDataset}
                loading={loading}
              />
            </aside>

            {/* Main Content */}
            <div className="main-content">

              {currentStep === 'upload' && (
                <div className="step-content">
                  <h2>Upload Your Dataset</h2>
                  <p>Upload a CSV file to generate privacy-safe synthetic data</p>
                  <FileUpload
                    onFileUpload={handleFileUpload}
                    loading={loading}
                  />
                </div>
              )}

              {currentStep === 'configure' && selectedDataset && (
                <div className="step-content">
                  <h2>Configure Privacy Settings</h2>
                  <p>Choose what data to mask and how to anonymize it</p>

                  {/* Dataset Info */}
                  <div className="dataset-info">
                    <h3>Dataset: {selectedDataset.filename}</h3>
                    <div className="dataset-stats">
                      <span>📊 {selectedDataset.row_count?.toLocaleString()} rows</span>
                      <span>📋 {selectedDataset.column_count} columns</span>
                      <span>💾 {(selectedDataset.file_size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                  </div>

                  {/* Data Preview */}
                  <DataPreview
                    datasetId={selectedDataset.id}
                    title="Original Data Preview"
                    synthetic={false}
                  />

                  {/* Privacy Configuration */}
                  <PrivacyConfig
                    dataset={selectedDataset}
                    onSubmit={handlePrivacyConfig}
                    loading={loading}
                  />
                </div>
              )}

              {currentStep === 'generate' && selectedDataset && (
                <div className="step-content">
                  <h2>Generating Synthetic Data</h2>
                  <p>Please wait while we generate privacy-safe synthetic data...</p>

                  <GenerationProgress
                    dataset={selectedDataset}
                    onComplete={() => setCurrentStep('review')}
                  />
                </div>
              )}

              {currentStep === 'review' && selectedDataset && (
                <div className="step-content">
                  <h2>Review Synthetic Data</h2>
                  <p>Review the quality and download your synthetic dataset</p>

                  {/* Action Buttons */}
                  <div className="action-buttons">
                    <button
                      className="btn btn-primary btn-large"
                      onClick={handleDownload}
                    >
                      📥 Download Synthetic Data
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setCurrentStep('configure')}
                    >
                      🔧 Reconfigure
                    </button>
                  </div>

                  {/* Data Previews */}
                  <div className="preview-grid">
                    <DataPreview
                      datasetId={selectedDataset.id}
                      title="Original Data Preview"
                      synthetic={false}
                    />
                    <DataPreview
                      datasetId={selectedDataset.id}
                      title="Synthetic Data Preview"
                      synthetic={true}
                    />
                  </div>

                  {/* Quality Report */}
                  {selectedDataset.quality_metrics && (
                    <QualityReport
                      qualityMetrics={selectedDataset.quality_metrics}
                      privacyConfig={selectedDataset.privacy_config}
                    />
                  )}
                </div>
              )}

            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <div className="container">
          <p>&copy; 2024 Synthetic Data Generator. Built with privacy and security in mind.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
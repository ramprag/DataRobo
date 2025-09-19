import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Components
import FileUpload from './components/FileUpload';
import DatasetList from './components/DatasetList';
import PrivacyConfig from './components/PrivacyConfig';
import GenerationProgress from './components/GenerationProgress';
import QualityReport from './components/QualityReport';
import DebugPanel from './components/DebugPanel';
import DataPreview from './components/DataPreview';
// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased timeout for file uploads
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [currentStep, setCurrentStep] = useState('upload'); // upload, configure, generate, review
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [pollIntervalRef, setPollIntervalRef] = useState(null);

  // Fetch datasets on component mount
  useEffect(() => {
    fetchDatasets();

    // Cleanup polling on unmount
    return () => {
      if (pollIntervalRef) {
        clearInterval(pollIntervalRef);
      }
    };
  }, []);

  // Cleanup polling when dataset changes
  useEffect(() => {
    return () => {
      if (pollIntervalRef) {
        clearInterval(pollIntervalRef);
        setPollIntervalRef(null);
      }
    };
  }, [selectedDataset?.id]);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/datasets');
      setDatasets(response.data || []);
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
      setError('Failed to fetch datasets: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for file upload
      });

      const newDataset = response.data;
      setDatasets(prev => [newDataset, ...prev]);
      setSelectedDataset(newDataset);
      setCurrentStep('configure');
      setSuccess('Dataset uploaded successfully!');

    } catch (err) {
      console.error('Upload failed:', err);
      const errorMessage = err.response?.data?.detail ||
                          err.response?.data?.message ||
                          err.message ||
                          'Unknown upload error';
      setError('Upload failed: ' + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (dataset) => {
    try {
      console.log('Dataset selection started for:', dataset.id, 'Status:', dataset.status);
      setError(null);
      setSuccess(null);

      // Clear any existing polling
      if (pollIntervalRef) {
        console.log('Clearing existing polling interval');
        clearInterval(pollIntervalRef);
        setPollIntervalRef(null);
      }

      // Fetch latest dataset status
      console.log('Fetching latest dataset status...');
      const response = await api.get(`/api/datasets/${dataset.id}`);
      const updatedDataset = response.data;

      console.log('Latest dataset status:', updatedDataset.status);
      setSelectedDataset(updatedDataset);

      // Set appropriate step based on status
      switch (updatedDataset.status) {
        case 'uploaded':
          console.log('Dataset is uploaded, going to configure step');
          setCurrentStep('configure');
          break;

        case 'processing':
          console.log('Dataset is processing, going to generate step and starting polling');
          setCurrentStep('generate');
          setLoading(true); // Set loading state for processing
          // Start polling for this dataset
          pollGenerationStatus(updatedDataset.id);
          break;

        case 'completed':
          console.log('Dataset is completed, going to review step');
          setCurrentStep('review');
          setLoading(false);
          setSuccess('Synthetic data is ready for review!');
          break;

        case 'failed':
          console.log('Dataset failed, going to configure step with error');
          setCurrentStep('configure');
          setLoading(false);
          setError(`Previous generation failed: ${updatedDataset.error_message || 'Unknown error'}`);
          break;

        default:
          console.log('Unknown status, defaulting to configure step');
          setCurrentStep('configure');
          setLoading(false);
      }
    } catch (err) {
      console.error('Failed to select dataset:', err);
      setError('Failed to load dataset details: ' + (err.response?.data?.detail || err.message));
      setSelectedDataset(dataset); // Set it anyway to allow retry
      setCurrentStep('configure');
      setLoading(false);
    }
  };

  const handlePrivacyConfig = async (privacyConfig, numRows = null) => {
    if (!selectedDataset) return;

    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      setCurrentStep('generate');

      const requestData = {
        dataset_id: selectedDataset.id,
        privacy_config: privacyConfig,
        num_rows: numRows
      };

      console.log('Sending generation request:', requestData);

      const response = await api.post(`/api/datasets/${selectedDataset.id}/generate-synthetic`, requestData);

      console.log('Generation request successful:', response.data);

      // Update dataset status immediately
      setSelectedDataset(prev => ({
        ...prev,
        status: 'processing',
        privacy_config: privacyConfig
      }));

      // Start polling for progress
      pollGenerationStatus(selectedDataset.id);

    } catch (err) {
      console.error('Failed to start generation:', err);
      const errorMessage = err.response?.data?.detail ||
                          err.response?.data?.message ||
                          err.message ||
                          'Unknown generation error';
      setError('Failed to start generation: ' + errorMessage);
      setCurrentStep('configure');
    } finally {
      setLoading(false);
    }
  };

  const pollGenerationStatus = (datasetId) => {
    // Clear any existing polling
    if (pollIntervalRef) {
      clearInterval(pollIntervalRef);
    }

    let pollCount = 0;
    const maxPolls = 150; // 5 minutes at 2-second intervals

    const intervalId = setInterval(async () => {
      pollCount++;

      if (pollCount > maxPolls) {
        clearInterval(intervalId);
        setPollIntervalRef(null);
        setError('Generation is taking longer than expected. Please refresh and check status.');
        setLoading(false);
        return;
      }

      try {
        const response = await api.get(`/api/datasets/${datasetId}`);
        const updatedDataset = response.data;

        console.log('Polling status:', updatedDataset.status, 'Poll count:', pollCount);

        setSelectedDataset(updatedDataset);

        // Update dataset in list
        setDatasets(prev =>
          prev.map(d => d.id === updatedDataset.id ? updatedDataset : d)
        );

        if (updatedDataset.status === 'completed') {
          clearInterval(intervalId);
          setPollIntervalRef(null);
          setCurrentStep('review');
          setSuccess('Synthetic data generated successfully!');
          setLoading(false);
        } else if (updatedDataset.status === 'failed') {
          clearInterval(intervalId);
          setPollIntervalRef(null);
          setCurrentStep('configure');
          setError(`Generation failed: ${updatedDataset.error_message || 'Unknown error occurred during generation'}`);
          setLoading(false);
        }

      } catch (err) {
        console.error('Error polling status:', err);
        if (pollCount > 3) { // Only show error after a few failed attempts
          clearInterval(intervalId);
          setPollIntervalRef(null);
          setError('Error checking generation status: ' + (err.response?.data?.detail || err.message));
          setLoading(false);
        }
      }
    }, 2000); // Poll every 2 seconds

    setPollIntervalRef(intervalId);
  };

  const handleDownload = async () => {
    if (!selectedDataset) return;

    try {
      setError(null);

      const response = await api.get(`/api/datasets/${selectedDataset.id}/download`, {
        responseType: 'blob',
        timeout: 60000, // 1 minute timeout for download
      });

      // Create download link
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedDataset.filename.replace(/\.[^/.]+$/, '')}_synthetic.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSuccess('Synthetic data downloaded successfully!');

    } catch (err) {
      console.error('Download failed:', err);
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
        // Clear polling if active
        if (pollIntervalRef) {
          clearInterval(pollIntervalRef);
          setPollIntervalRef(null);
        }
      }

      setSuccess('Dataset deleted successfully!');

    } catch (err) {
      console.error('Delete failed:', err);
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
                loading={loading && datasets.length === 0}
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
                      <span>📊 {selectedDataset.row_count?.toLocaleString() || 'Unknown'} rows</span>
                      <span>📋 {selectedDataset.column_count || 'Unknown'} columns</span>
                      <span>💾 {selectedDataset.file_size ? (selectedDataset.file_size / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown size'}</span>
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
                      dataset={selectedDataset}
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

      {/* Debug Panel - only shows in development */}
      <DebugPanel
        selectedDataset={selectedDataset}
        currentStep={currentStep}
        loading={loading}
        error={error}
        success={success}
      />
    </div>
  );
}

export default App;
// frontend/src/App.js
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
import RelationshipViewer from './components/RelationshipViewer';
import ImportExportModal from './components/ImportExportModal';
import AIFabricate from './components/AIFabricate';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // Increased to 5 minutes
  headers: {
    'Content-Type': 'application/json',
  }
});

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
  const [currentStep, setCurrentStep] = useState('upload');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [pollIntervalRef, setPollIntervalRef] = useState(null);

  // New states for Import/Export and AI Fabricate
  const [showImportModal, setShowImportModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [generationMode, setGenerationMode] = useState('schema'); // 'schema' or 'ai'

  // CI/CD states
  const [showCICDModal, setShowCICDModal] = useState(false);
  const [pipelines, setPipelines] = useState([]);
  const [pipelineRuns, setPipelineRuns] = useState([]);
  const [selectedPipeline, setSelectedPipeline] = useState(null);

  useEffect(() => {
    fetchDatasets();

    return () => {
      if (pollIntervalRef) {
        clearInterval(pollIntervalRef);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (pollIntervalRef) {
        clearInterval(pollIntervalRef);
        setPollIntervalRef(null);
      }
    };
  }, [selectedDataset?.id]);

  // Add useEffect to fetch CI/CD data when modal opens
  useEffect(() => {
    if (showCICDModal) {
      fetchPipelines();
      fetchPipelineRuns();
    }
  }, [showCICDModal]);

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

      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      const allowedTypes = ['.csv', '.xlsx', '.xls', '.zip'];

      if (!allowedTypes.includes(fileExtension)) {
        throw new Error('Invalid file type. Please upload CSV, Excel, or ZIP files only.');
      }

      const maxSize = fileExtension === '.zip' ? 100 * 1024 * 1024 : 50 * 1024 * 1024;
      if (file.size > maxSize) {
        const maxSizeLabel = fileExtension === '.zip' ? '100MB' : '50MB';
        throw new Error(`File size exceeds maximum of ${maxSizeLabel}`);
      }

      console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', fileExtension);

      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log('Upload progress:', percentCompleted + '%');
        }
      });

      const newDataset = response.data;

      if (!newDataset || !newDataset.id) {
        throw new Error('Invalid response from server');
      }

      setDatasets(prev => [newDataset, ...prev]);
      setSelectedDataset(newDataset);
      setCurrentStep('configure');

      const isZip = fileExtension === '.zip';
      const successMsg = isZip
        ? `✅ Multi-table dataset uploaded! ${newDataset.table_count || 0} tables detected with ${newDataset.relationship_summary?.total_relationships || 0} relationships.`
        : '✅ Dataset uploaded successfully!';

      setSuccess(successMsg);
      console.log('Upload successful:', newDataset);

    } catch (err) {
      console.error('Upload failed:', err);

      let errorMessage = 'Upload failed';

      if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      } else {
        errorMessage = 'Unknown upload error occurred';
      }

      setError('❌ ' + errorMessage);
      setCurrentStep('upload');

    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (dataset) => {
    try {
      console.log('Dataset selection started for:', dataset.id, 'Status:', dataset.status);
      setError(null);
      setSuccess(null);

      if (pollIntervalRef) {
        console.log('Clearing existing polling interval');
        clearInterval(pollIntervalRef);
        setPollIntervalRef(null);
      }

      console.log('Fetching latest dataset status...');
      const response = await api.get(`/api/datasets/${dataset.id}`, {
        timeout: 30000 // 30 seconds for status check (more tolerant)
      });
      const updatedDataset = response.data;

      console.log('Latest dataset status:', updatedDataset.status);
      setSelectedDataset(updatedDataset);

      switch (updatedDataset.status) {
        case 'uploaded':
          console.log('Dataset is uploaded, going to configure step');
          setCurrentStep('configure');
          break;

        case 'processing':
          console.log('Dataset is processing, going to generate step and starting polling');
          setCurrentStep('generate');
          setLoading(true);
          pollGenerationStatus(updatedDataset.id, updatedDataset.progress || 0);
          break;

        case 'completed':
          console.log('Dataset is completed, going to review step');
          setCurrentStep('review');
          setLoading(false);
          setSuccess('✅ Synthetic data is ready for review!');
          break;

        case 'failed':
          console.log('Dataset failed, going to configure step with error');
          setCurrentStep('configure');
          setLoading(false);
          setError(`❌ Previous generation failed: ${updatedDataset.error_message || 'Unknown error'}`);
          break;

        default:
          console.log('Unknown status, defaulting to configure step');
          setCurrentStep('configure');
          setLoading(false);
      }
    } catch (err) {
      console.error('Failed to select dataset:', err);
      setError('Failed to load dataset details: ' + (err.response?.data?.detail || err.message));
      setSelectedDataset(dataset);
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

      const response = await api.post(`/api/datasets/${selectedDataset.id}/generate-synthetic`, requestData, {
        timeout: 60000 // Just to start the process
      });

      console.log('Generation request successful:', response.data);

      setSelectedDataset(prev => ({
        ...prev,
        status: 'processing',
        privacy_config: privacyConfig,
        progress: 0
      }));

      pollGenerationStatus(selectedDataset.id, 0);

    } catch (err) {
      console.error('Failed to start generation:', err);
      const errorMessage = err.response?.data?.detail ||
                          err.response?.data?.message ||
                          err.message ||
                          'Unknown generation error';
      setError('❌ Failed to start generation: ' + errorMessage);
      setCurrentStep('configure');
    } finally {
      setLoading(false);
    }
  };

  // More resilient polling: higher timeout, more consecutive error tolerance,
  // final confirmation check before giving up, and continued polling when near completion
  const pollGenerationStatus = (datasetId, startingProgress = 0) => {
    if (pollIntervalRef) {
      clearInterval(pollIntervalRef);
    }

    let pollCount = 0;
    const baseIntervalMs = 3000;           // 3s poll interval
    const baseMaxPolls = 600;              // ~30 minutes at 3s
    let maxPolls = baseMaxPolls;
    let consecutiveErrors = 0;
    let lastKnownProgress = startingProgress || 0;
    let lastKnownStatus = 'processing';

    const stopPolling = () => {
      clearInterval(intervalId);
      setPollIntervalRef(null);
    };

    const confirmCompletion = async () => {
      try {
        const finalResp = await api.get(`/api/datasets/${datasetId}`, { timeout: 60000 });
        const finalData = finalResp.data;
        if (finalData.status === 'completed') {
          stopPolling();
          setSelectedDataset(finalData);
          setDatasets(prev => prev.map(d => d.id === finalData.id ? finalData : d));
          setCurrentStep('review');
          setSuccess('✅ Synthetic data generated successfully!');
          setLoading(false);
          return true;
        }
      } catch (e) {
        // ignore; we'll handle below
      }
      return false;
    };

    const intervalId = setInterval(async () => {
      pollCount++;

      // Be extra patient near completion
      if (lastKnownProgress >= 90) {
        maxPolls = Math.max(maxPolls, baseMaxPolls * 2); // up to ~60 minutes if needed
      }

      if (pollCount > maxPolls) {
        // Final confirmation before giving up
        confirmCompletion().then((done) => {
          if (done) return;
          stopPolling();
          if (lastKnownProgress >= 90) {
            setError('⚠️ Connection issues while finalizing. Data may be ready—please refresh or try again in a moment.');
          } else {
            setError('⚠️ Generation is taking longer than expected. Please refresh and check status.');
          }
          setLoading(false);
        });
        return;
      }

      try {
        const response = await api.get(`/api/datasets/${datasetId}`, {
          timeout: 60000 // 60s tolerance for slower environments
        });
        const updatedDataset = response.data;

        consecutiveErrors = 0;

        lastKnownProgress = typeof updatedDataset.progress === 'number'
          ? updatedDataset.progress
          : lastKnownProgress;
        lastKnownStatus = updatedDataset.status || lastKnownStatus;

        setSelectedDataset(updatedDataset);
        setDatasets(prev => prev.map(d => d.id === updatedDataset.id ? updatedDataset : d));

        if (updatedDataset.status === 'completed') {
          stopPolling();
          setCurrentStep('review');
          setSuccess('✅ Synthetic data generated successfully!');
          setLoading(false);
        } else if (updatedDataset.status === 'failed') {
          stopPolling();
          setCurrentStep('configure');
          setError(`❌ Generation failed: ${updatedDataset.error_message || 'Unknown error occurred during generation'}`);
          setLoading(false);
        }

      } catch (err) {
        const msg = (typeof err.message === 'string' ? err.message.toLowerCase() : '');
        const isTimeout = err.code === 'ECONNABORTED' || msg.includes('timeout');

        consecutiveErrors += 1;

        // Near completion: allow many transient timeouts without scaring the user
        const nearCompletion = lastKnownProgress >= 90 || lastKnownStatus === 'processing';
        const timeoutOnlyThreshold = nearCompletion ? 20 : 10; // soft tolerance for timeouts
        const hardErrorThreshold = nearCompletion ? 12 : 6;     // total consecutive errors of any type

        if (isTimeout && consecutiveErrors < timeoutOnlyThreshold) {
          // Silent continue to avoid flashing an error
          return;
        }

        if (consecutiveErrors >= hardErrorThreshold) {
          // Final confirmation before surfacing error
          confirmCompletion().then((done) => {
            if (done) return;
            stopPolling();
            if (isTimeout) {
              // Show softer guidance—generation likely succeeded but polling struggled
              setError('⚠️ Connection timeout. Generation may still be running. Please refresh to check status.');
            } else {
              setError('❌ Error checking generation status. Please refresh to check status.');
            }
            setLoading(false);
          });
        }
      }
    }, baseIntervalMs);

    setPollIntervalRef(intervalId);
  };

  const handleDownload = async () => {
    if (!selectedDataset) return;

    try {
      setError(null);

      const response = await api.get(`/api/datasets/${selectedDataset.id}/download`, {
        responseType: 'blob',
        timeout: 120000, // 2 minutes
      });

      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedDataset.filename.replace(/\.[^/.]+$/, '')}_synthetic.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSuccess('✅ Synthetic data downloaded successfully!');

    } catch (err) {
      console.error('Download failed:', err);
      setError('❌ Download failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDownloadMultiTable = async () => {
    if (!selectedDataset) return;

    try {
      setError(null);

      const response = await api.get(`/api/datasets/${selectedDataset.id}/download-zip`, {
        responseType: 'blob',
        timeout: 120000, // 2 minutes
      });

      const blob = new Blob([response.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedDataset.filename.replace(/\.[^/.]+$/, '')}_synthetic.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSuccess('✅ Multi-table synthetic data downloaded successfully!');

    } catch (err) {
      console.error('Multi-table download failed:', err);
      setError('❌ Download failed: ' + (err.response?.data?.detail || err.message));
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
        if (pollIntervalRef) {
          clearInterval(pollIntervalRef);
          setPollIntervalRef(null);
        }
      }

      setSuccess('✅ Dataset deleted successfully!');

    } catch (err) {
      console.error('Delete failed:', err);
      setError('❌ Delete failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  // New handlers for Import/Export
  const handleImport = (source, file) => {
    if (source === 'local' && file) {
      handleFileUpload(file);
      setShowImportModal(false);
    } else {
      alert(`${source} import will be implemented soon!`);
    }
  };

  const handleExport = (destination) => {
    if (!selectedDataset) {
      alert('Please select a dataset first');
      return;
    }

    if (destination === 'local') {
      if (selectedDataset.table_count > 1) {
        handleDownloadMultiTable();
      } else {
        handleDownload();
      }
      setShowExportModal(false);
    } else {
      alert(`${destination} export will be implemented soon!`);
    }
  };

  // CI/CD Pipeline handlers
  const handleCreatePipeline = async (pipelineData) => {
    try {
      setLoading(true);
      const response = await api.post('/api/cicd/pipelines', pipelineData);
      const newPipeline = response.data;
      setPipelines(prev => [newPipeline, ...prev]);
      setSuccess('✅ Pipeline created successfully!');
      setShowCICDModal(false);
    } catch (err) {
      console.error('Failed to create pipeline:', err);
      setError('❌ Failed to create pipeline: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleRunPipeline = async (pipelineId, datasetId) => {
    try {
      setLoading(true);
      const response = await api.post(`/api/cicd/pipelines/${pipelineId}/run`, {
        dataset_id: datasetId
      });
      const newRun = response.data;
      setPipelineRuns(prev => [newRun, ...prev]);
      setSuccess('✅ Pipeline run started successfully!');
    } catch (err) {
      console.error('Failed to run pipeline:', err);
      setError('❌ Failed to run pipeline: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchPipelines = async () => {
    try {
      const response = await api.get('/api/cicd/pipelines');
      setPipelines(response.data || []);
    } catch (err) {
      console.error('Failed to fetch pipelines:', err);
    }
  };

  const fetchPipelineRuns = async () => {
    try {
      const response = await api.get('/api/cicd/runs');
      setPipelineRuns(response.data || []);
    } catch (err) {
      console.error('Failed to fetch pipeline runs:', err);
    }
  };

  // Handler for AI Fabricate
  const handleAIFabricate = (prompt, config) => {
    console.log('AI Fabricate request:', prompt, config);
    alert('AI Fabricate feature coming soon! Our team is integrating advanced LLM capabilities.');
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h1>🔒 GSTAN - Synthetic Data Generator</h1>
              <p>Privacy-safe synthetic data generation with GAN support</p>
            </div>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <button
                onClick={() => setShowImportModal(true)}
                className="btn"
                style={{
                  background: 'white',
                  color: '#667eea',
                  border: '2px solid white',
                  fontWeight: '600'
                }}
              >
                📥 Import Data
              </button>
              <button
                onClick={() => setShowExportModal(true)}
                className="btn"
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  border: '2px solid white',
                  fontWeight: '600'
                }}
                disabled={!selectedDataset}
              >
                📤 Export Data
              </button>
              <button
                onClick={() => setShowCICDModal(true)}
                className="btn"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  color: 'white',
                  border: '2px solid rgba(255, 255, 255, 0.3)',
                  fontWeight: '600'
                }}
              >
                🔄 CI/CD Pipeline
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">

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

          <div className="app-content">
            <aside className="sidebar">
              <DatasetList
                datasets={datasets}
                selectedDataset={selectedDataset}
                onDatasetSelect={handleDatasetSelect}
                onDatasetDelete={handleDeleteDataset}
                loading={loading && datasets.length === 0}
              />
            </aside>

            <div className="main-content">

              {currentStep === 'upload' && (
                <div className="step-content">
                  <h2>Welcome to GSTAN</h2>
                  <p>Choose your data generation method</p>

{/* Generation Mode Selector - Now at the top */}
<div className="mode-switch">
  <button
    type="button"
    onClick={() => setGenerationMode('schema')}
    className={`mode-card ${generationMode === 'schema' ? 'active' : ''}`}
    aria-pressed={generationMode === 'schema'}
  >
    <div className="icon">📊</div>
    <div className="content">
      <div className="title">Schema-Based Generation</div>
      <div className="subtitle">Upload your dataset (CSV/Excel or multi-table ZIP)</div>
      <div className="meta">
        <span>Relationship detection</span>
        <span>•</span>
        <span>Privacy masking</span>
        <span>•</span>
        <span>GAN optional</span>
      </div>
    </div>
  </button>

  <button
    type="button"
    onClick={() => setGenerationMode('ai')}
    className={`mode-card ${generationMode === 'ai' ? 'active' : ''}`}
    aria-pressed={generationMode === 'ai'}
  >
    <span className="badge">BETA</span>
    <div className="icon">✨</div>
    <div className="content">
      <div className="title">AI Fabricate</div>
      <div className="subtitle">Describe your data and we'll generate it</div>
      <div className="meta">
        <span>Natural language</span>
        <span>•</span>
        <span>Enterprise-ready</span>
        <span>•</span>
        <span>Multi-domain</span>
      </div>
    </div>
  </button>
</div>
                  {generationMode === 'schema' ? (
                    <>
                      <h3 style={{ marginBottom: '1rem', color: '#1e293b' }}>Upload Your Dataset</h3>
                      <p style={{ marginBottom: '2rem', color: '#64748b' }}>Upload a CSV file, Excel file, or ZIP archive of multiple tables</p>
                      <FileUpload
                        onFileUpload={handleFileUpload}
                        loading={loading}
                      />
                    </>
                  ) : (
                    <>
                      <h3 style={{ marginBottom: '1rem', color: '#a855f7' }}>AI-Powered Data Generation</h3>
                      <p style={{ marginBottom: '2rem', color: '#64748b' }}>Describe your dataset and let AI create it for you</p>
                      <AIFabricate
                        onGenerate={handleAIFabricate}
                        loading={loading}
                      />
                    </>
                  )}
                </div>
              )}

              {currentStep === 'configure' && selectedDataset && (
                <div className="step-content">
                  <h2>Configure Privacy Settings</h2>
                  <p>Choose what data to mask and select generation method</p>

                  {/* Progress indicator for current step */}
                  <div className="progress-steps-inline" style={{ marginBottom: '2rem' }}>
                    <div className="step-indicator" style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.75rem 1.5rem',
                      background: '#eff6ff',
                      borderRadius: '8px',
                      border: '2px solid #3b82f6',
                      width: 'fit-content'
                    }}>
                      <span style={{ fontSize: '1.5rem' }}>⚙️</span>
                      <span style={{ fontWeight: '600', color: '#1e40af' }}>Step 2: Configure Privacy</span>
                    </div>
                  </div>

                  <div className="dataset-info">
                    <h3>Dataset: {selectedDataset.filename}</h3>
                    <div className="dataset-stats">
                      {selectedDataset.table_count > 1 && (
                        <span>📦 {selectedDataset.table_count} tables</span>
                      )}
                      <span>📊 {selectedDataset.row_count?.toLocaleString() || 'Unknown'} total rows</span>
                      <span>📋 {selectedDataset.column_count || 'Unknown'} total columns</span>
                      <span>💾 {selectedDataset.file_size ? (selectedDataset.file_size / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown size'}</span>
                      {selectedDataset.relationship_summary?.total_relationships > 0 && (
                        <span>🔗 {selectedDataset.relationship_summary.total_relationships} relationships</span>
                      )}
                    </div>
                  </div>

                  {selectedDataset.table_count > 1 && (
                    <RelationshipViewer
                      datasetId={selectedDataset.id}
                      relationshipData={selectedDataset}
                    />
                  )}

                  <DataPreview
                    datasetId={selectedDataset.id}
                    title="Original Data Preview"
                    synthetic={false}
                  />

                  <PrivacyConfig
                    dataset={selectedDataset}
                    onSubmit={handlePrivacyConfig}
                    loading={loading}
                  />
                </div>
              )}

              {currentStep === 'generate' && selectedDataset && (
                <div className="step-content">
                  {/* Progress indicator for current step */}
                  <div className="progress-steps-inline" style={{ marginBottom: '2rem' }}>
                    <div className="step-indicator" style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.75rem 1.5rem',
                      background: '#fef3c7',
                      borderRadius: '8px',
                      border: '2px solid #f59e0b',
                      width: 'fit-content'
                    }}>
                      <span style={{ fontSize: '1.5rem' }}>🔄</span>
                      <span style={{ fontWeight: '600', color: '#92400e' }}>Step 3: Generating Synthetic Data</span>
                    </div>
                  </div>

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
                  {/* Progress indicator for current step */}
                  <div className="progress-steps-inline" style={{ marginBottom: '2rem' }}>
                    <div className="step-indicator" style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.75rem 1.5rem',
                      background: '#d1fae5',
                      borderRadius: '8px',
                      border: '2px solid #10b981',
                      width: 'fit-content'
                    }}>
                      <span style={{ fontSize: '1.5rem' }}>✅</span>
                      <span style={{ fontWeight: '600', color: '#065f46' }}>Step 4: Review & Download</span>
                    </div>
                  </div>

                  <h2>Review Synthetic Data</h2>
                  <p>Review the quality and download your synthetic dataset</p>

                  <div className="action-buttons">
                    {selectedDataset.table_count > 1 ? (
                      <button
                        className="btn btn-primary btn-large"
                        onClick={handleDownloadMultiTable}
                        style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
                      >
                        📦 Download All Tables (ZIP)
                      </button>
                    ) : (
                      <button
                        className="btn btn-primary btn-large"
                        onClick={handleDownload}
                      >
                        📥 Download Synthetic Data
                      </button>
                    )}

                    <button
                      className="btn btn-secondary"
                      onClick={() => setCurrentStep('configure')}
                    >
                      🔧 Reconfigure
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        setCurrentStep('upload');
                        setSelectedDataset(null);
                        if (pollIntervalRef) {
                          clearInterval(pollIntervalRef);
                          setPollIntervalRef(null);
                        }
                      }}
                    >
                      🏠 Back to Home
                    </button>
                  </div>

                  {selectedDataset.table_count > 1 && (
                    <RelationshipViewer
                      datasetId={selectedDataset.id}
                      relationshipData={selectedDataset}
                    />
                  )}

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
          <p>&copy; 2024 GSTAN Synthetic Data Generator. Built with privacy and security in mind.</p>
        </div>
      </footer>

      {/* Import/Export Modals */}
      <ImportExportModal
        show={showImportModal}
        mode="import"
        onClose={() => setShowImportModal(false)}
        onSubmit={handleImport}
      />

      <ImportExportModal
        show={showExportModal}
        mode="export"
        onClose={() => setShowExportModal(false)}
        onSubmit={handleExport}
        selectedDataset={selectedDataset}
      />

      {/* CI/CD Pipeline Modal */}
{/* Enterprise CI/CD Pipeline Modal */}
{showCICDModal && (
  <div className="modal-overlay enterprise-modal">
    <div className="modal-content enterprise-cicd-modal">
      <div className="enterprise-modal-header">
        <div className="header-content">
          <div className="header-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="header-text">
            <h2>CI/CD Pipeline Management</h2>
            <p>Automate synthetic data generation with enterprise-grade quality gates</p>
          </div>
        </div>
        <button onClick={() => setShowCICDModal(false)} className="enterprise-close-btn">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 5L5 15M5 5L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      <div className="enterprise-cicd-content">
        <div className="enterprise-tabs">
          <button
            className={`enterprise-tab ${selectedPipeline === null ? 'active' : ''}`}
            onClick={() => setSelectedPipeline(null)}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M2 3H14V13H2V3Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M6 7H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M6 9H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <span>Pipeline Library</span>
            <div className="tab-badge">{pipelines.length}</div>
          </button>
          <button
            className={`enterprise-tab ${selectedPipeline === 'runs' ? 'active' : ''}`}
            onClick={() => setSelectedPipeline('runs')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M8 1V15M1 8H15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
            <span>Execution History</span>
            <div className="tab-badge">{pipelineRuns.length}</div>
          </button>
          <button
            className={`enterprise-tab ${selectedPipeline === 'create' ? 'active' : ''}`}
            onClick={() => setSelectedPipeline('create')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M8 1V15M1 8H15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <span>Create Pipeline</span>
          </button>
        </div>

        <div className="enterprise-content-area">
          {selectedPipeline === null && (
            <div className="enterprise-section">
              <div className="section-header-enterprise">
                <div className="header-left">
                  <h3>Pipeline Library</h3>
                  <p>Manage your automated data generation pipelines</p>
                </div>
                <div className="header-actions">
                  <button
                    className="enterprise-btn enterprise-btn-primary"
                    onClick={() => setSelectedPipeline('create')}
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M8 1V15M1 8H15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                    Create New Pipeline
                  </button>
                </div>
              </div>

              <div className="enterprise-pipelines-grid">
                {pipelines.map(pipeline => (
                  <div key={pipeline.id} className="enterprise-pipeline-card">
                    <div className="pipeline-card-header">
                      <div className="pipeline-info">
                        <h4>{pipeline.name}</h4>
                        <p>{pipeline.description}</p>
                      </div>
                      <div className="pipeline-status">
                        <span className={`enterprise-status-badge ${pipeline.active ? 'active' : 'inactive'}`}>
                          {pipeline.active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                    </div>

                    <div className="pipeline-metrics-enterprise">
                      <div className="metric">
                        <div className="metric-icon">🎯</div>
                        <div className="metric-content">
                          <span className="metric-label">Quality Gate</span>
                          <span className="metric-value">{pipeline.quality_gate?.min_overall_quality || 60}%</span>
                        </div>
                      </div>
                      <div className="metric">
                        <div className="metric-icon">⚡</div>
                        <div className="metric-content">
                          <span className="metric-label">Auto-trigger</span>
                          <span className="metric-value">{pipeline.auto_trigger_on_upload ? 'ON' : 'OFF'}</span>
                        </div>
                      </div>
                    </div>

                    <div className="pipeline-actions-enterprise">
                      <button
                        className="enterprise-btn enterprise-btn-secondary"
                        onClick={() => handleRunPipeline(pipeline.id, selectedDataset?.id)}
                        disabled={!selectedDataset}
                      >
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M3 2L13 8L3 14V2Z" fill="currentColor"/>
                        </svg>
                        Execute Pipeline
                      </button>
                      <button className="enterprise-btn enterprise-btn-outline">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2H4C2.9 2 2 2.9 2 4V12C2 13.1 2.9 14 4 14H12C13.1 14 14 13.1 14 12V4C14 2.9 13.1 2 12 2Z" stroke="currentColor" strokeWidth="1.5"/>
                          <path d="M6 6H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                          <path d="M6 8H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                        </svg>
                        Configure
                      </button>
                    </div>
                  </div>
                ))}
                {pipelines.length === 0 && (
                  <div className="enterprise-empty-state">
                    <div className="empty-icon">
                      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M24 4L44 14L24 24L4 14L24 4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M4 28L24 38L44 28" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M4 18L24 28L44 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                    <h4>No Pipelines Created</h4>
                    <p>Create your first pipeline to automate synthetic data generation with quality gates</p>
                    <button
                      className="enterprise-btn enterprise-btn-primary"
                      onClick={() => setSelectedPipeline('create')}
                    >
                      Create Your First Pipeline
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {selectedPipeline === 'create' && (
            <div className="enterprise-section">
              <div className="section-header-enterprise">
                <div className="header-left">
                  <h3>Create New Pipeline</h3>
                  <p>Configure an enterprise-grade automated data generation pipeline</p>
                </div>
                <div className="header-actions">
                  <button
                    className="enterprise-btn enterprise-btn-secondary"
                    onClick={() => setSelectedPipeline(null)}
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                    Cancel
                  </button>
                </div>
              </div>

              <div className="enterprise-form-container">
                <form onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.target);
                  const pipelineData = {
                    name: formData.get('name'),
                    description: formData.get('description'),
                    auto_trigger_on_upload: formData.get('auto_trigger') === 'on',
                    default_privacy: {
                      mask_emails: formData.get('mask_emails') === 'on',
                      mask_names: formData.get('mask_names') === 'on',
                      mask_phone_numbers: formData.get('mask_phone_numbers') === 'on',
                      mask_addresses: formData.get('mask_addresses') === 'on',
                      mask_ssn: formData.get('mask_ssn') === 'on',
                      custom_fields: [],
                      use_gan: formData.get('use_gan') === 'on',
                      gan_epochs: parseInt(formData.get('gan_epochs') || '100'),
                      anonymization_method: 'faker'
                    },
                    quality_gate: {
                      min_overall_quality: parseFloat(formData.get('min_quality') || '60'),
                      allow_missing_columns: formData.get('allow_missing_columns') === 'on'
                    },
                    export_target: {
                      type: formData.get('export_type') || 'local',
                      path: formData.get('export_path') || 'artifacts'
                    },
                    active: true
                  };
                  handleCreatePipeline(pipelineData);
                }}>
                  <div className="form-sections">
                    <div className="form-section">
                      <div className="section-header">
                        <h4>Basic Configuration</h4>
                        <p>Define your pipeline name and behavior</p>
                      </div>
                      <div className="form-grid">
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">Pipeline Name</span>
                            <span className="label-required">*</span>
                          </label>
                          <input
                            type="text"
                            name="name"
                            required
                            className="enterprise-input"
                            placeholder="e.g., Production Quality Pipeline"
                          />
                        </div>
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">Description</span>
                            <span className="label-required">*</span>
                          </label>
                          <textarea
                            name="description"
                            required
                            className="enterprise-textarea"
                            placeholder="Describe what this pipeline validates and its purpose..."
                          ></textarea>
                        </div>
                        <div className="form-group-enterprise">
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="auto_trigger" />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Auto-trigger on dataset upload</span>
                              <span className="checkbox-description">Automatically execute this pipeline when new datasets are uploaded</span>
                            </div>
                          </label>
                        </div>
                      </div>
                    </div>

                    <div className="form-section">
                      <div className="section-header">
                        <h4>Privacy Configuration</h4>
                        <p>Configure data masking and anonymization settings</p>
                      </div>
                      <div className="form-grid">
                        <div className="privacy-grid">
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="mask_emails" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Mask Email Addresses</span>
                            </div>
                          </label>
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="mask_names" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Mask Names</span>
                            </div>
                          </label>
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="mask_phone_numbers" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Mask Phone Numbers</span>
                            </div>
                          </label>
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="mask_addresses" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Mask Addresses</span>
                            </div>
                          </label>
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="mask_ssn" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Mask SSN</span>
                            </div>
                          </label>
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="use_gan" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Use GAN Generation</span>
                            </div>
                          </label>
                        </div>
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">GAN Training Epochs</span>
                          </label>
                          <input
                            type="number"
                            name="gan_epochs"
                            min="10"
                            max="500"
                            defaultValue="100"
                            className="enterprise-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="form-section">
                      <div className="section-header">
                        <h4>Quality Gates</h4>
                        <p>Set minimum quality thresholds for pipeline success</p>
                      </div>
                      <div className="form-grid">
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">Minimum Overall Quality Score</span>
                            <span className="label-required">*</span>
                          </label>
                          <div className="input-with-unit">
                            <input
                              type="number"
                              name="min_quality"
                              min="0"
                              max="100"
                              defaultValue="60"
                              className="enterprise-input"
                              placeholder="60"
                            />
                            <span className="input-unit">%</span>
                          </div>
                          <div className="form-help">Pipeline will fail if synthetic data quality falls below this threshold</div>
                        </div>
                        <div className="form-group-enterprise">
                          <label className="enterprise-checkbox">
                            <input type="checkbox" name="allow_missing_columns" defaultChecked />
                            <span className="checkmark"></span>
                            <div className="checkbox-content">
                              <span className="checkbox-label">Allow Missing Columns</span>
                              <span className="checkbox-description">Permit generation even if some columns are missing</span>
                            </div>
                          </label>
                        </div>
                      </div>
                    </div>

                    <div className="form-section">
                      <div className="section-header">
                        <h4>Export Configuration</h4>
                        <p>Configure where generated artifacts are stored</p>
                      </div>
                      <div className="form-grid">
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">Export Type</span>
                          </label>
                          <select name="export_type" className="enterprise-select" defaultValue="local">
                            <option value="local">Local Storage</option>
                            <option value="s3">Amazon S3</option>
                            <option value="gcs">Google Cloud Storage</option>
                            <option value="azure">Azure Blob Storage</option>
                          </select>
                        </div>
                        <div className="form-group-enterprise">
                          <label className="enterprise-label">
                            <span className="label-text">Export Path/Bucket</span>
                          </label>
                          <input
                            type="text"
                            name="export_path"
                            className="enterprise-input"
                            defaultValue="artifacts"
                            placeholder="artifacts"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="form-actions-enterprise">
                    <button type="submit" className="enterprise-btn enterprise-btn-primary enterprise-btn-large">
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13.5 2L6 9.5L2.5 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      Create Pipeline
                    </button>
                    <button
                      type="button"
                      className="enterprise-btn enterprise-btn-secondary"
                      onClick={() => setSelectedPipeline(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {selectedPipeline === 'runs' && (
            <div className="enterprise-section">
              <div className="section-header-enterprise">
                <div className="header-left">
                  <h3>Execution History</h3>
                  <p>Monitor pipeline runs and their performance metrics</p>
                </div>
              </div>

              <div className="enterprise-runs-container">
                {pipelineRuns.map(run => (
                  <div key={run.id} className="enterprise-run-card">
                    <div className="run-card-header">
                      <div className="run-info">
                        <div className="run-id">#{run.id.slice(0, 8)}</div>
                        <div className="run-details">
                          <span>Pipeline: {run.pipeline_id?.slice(0, 8)}</span>
                          <span>Dataset: {run.dataset_id?.slice(0, 8)}</span>
                        </div>
                      </div>
                      <div className="run-status-container">
                        <span className={`enterprise-run-status ${run.status}`}>
                          {run.status}
                        </span>
                        <div className="run-time">
                          {new Date(run.started_at).toLocaleString()}
                        </div>
                      </div>
                    </div>

                    {run.progress > 0 && (
                      <div className="run-progress-container">
                        <div className="progress-header">
                          <span>Progress</span>
                          <span>{run.progress}%</span>
                        </div>
                        <div className="enterprise-progress-bar">
                          <div
                            className="enterprise-progress-fill"
                            style={{ width: `${run.progress}%` }}
                          ></div>
                        </div>
                      </div>
                    )}

                    {run.message && (
                      <div className="run-message-container">
                        <span className="run-message">{run.message}</span>
                      </div>
                    )}
                  </div>
                ))}
                {pipelineRuns.length === 0 && (
                  <div className="enterprise-empty-state">
                    <div className="empty-icon">
                      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 1V15M1 8H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="2"/>
                      </svg>
                    </div>
                    <h4>No Pipeline Runs</h4>
                    <p>Execute a pipeline to see run history and performance metrics here</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
)}

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
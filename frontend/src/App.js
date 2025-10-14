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
        timeout: 15000 // Just to start the process
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
// Replace your existing pollGenerationStatus in frontend/src/App.js with this version
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
      <div className="subtitle">Describe your data and we’ll generate it</div>
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
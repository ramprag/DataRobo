import React, { useState, useRef } from 'react';
import './FileUpload.css';

const FileUpload = ({ onFileUpload, loading }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleFileSelect = (file) => {
    // Validate file type
    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!allowedTypes.includes(fileExtension)) {
      alert('Please select a CSV or Excel file (.csv, .xlsx, .xls)');
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      alert('File size must be less than 100MB');
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile && onFileUpload) {
      onFileUpload(selectedFile);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="file-upload">
      {!selectedFile ? (
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <h3>Drop your dataset here</h3>
            <p>or <span className="upload-link">click to browse</span></p>
            <div className="upload-formats">
              Supported formats: CSV, Excel (.csv, .xlsx, .xls)
            </div>
            <div className="upload-limit">
              Maximum file size: 100MB
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="file-selected">
          <div className="file-info">
            <div className="file-icon">📄</div>
            <div className="file-details">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-meta">
                <span>{formatFileSize(selectedFile.size)}</span>
                <span>•</span>
                <span>{selectedFile.type || 'Unknown type'}</span>
              </div>
            </div>
            <button
              className="btn btn-text"
              onClick={handleClearFile}
              disabled={loading}
            >
              ✕
            </button>
          </div>

          <div className="upload-actions">
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Uploading...
                </>
              ) : (
                '📤 Upload Dataset'
              )}
            </button>
            <button
              className="btn btn-secondary"
              onClick={handleClearFile}
              disabled={loading}
            >
              Choose Different File
            </button>
          </div>
        </div>
      )}

      {/* Sample Data Section */}
      <div className="sample-data-section">
        <h4>Don't have data? Try our sample datasets:</h4>
        <div className="sample-datasets">
          <button
            className="btn btn-outline"
            onClick={() => {
              // Create a sample CSV file
              const sampleData = `name,email,age,city,salary
John Doe,john.doe@email.com,28,New York,75000
Jane Smith,jane.smith@email.com,32,Los Angeles,85000
Mike Johnson,mike.johnson@email.com,25,Chicago,65000
Sarah Wilson,sarah.wilson@email.com,29,Houston,72000
David Brown,david.brown@email.com,35,Phoenix,90000`;

              const blob = new Blob([sampleData], { type: 'text/csv' });
              const file = new File([blob], 'sample_employees.csv', { type: 'text/csv' });
              handleFileSelect(file);
            }}
            disabled={loading}
          >
            👥 Employee Data Sample
          </button>

          <button
            className="btn btn-outline"
            onClick={() => {
              // Create a sample customer CSV file
              const sampleData = `customer_id,first_name,last_name,email,phone,address,purchase_amount,registration_date
C001,Alice,Johnson,alice.j@email.com,555-0101,123 Main St,1250.50,2023-01-15
C002,Bob,Williams,bob.w@email.com,555-0102,456 Oak Ave,890.75,2023-02-20
C003,Carol,Davis,carol.d@email.com,555-0103,789 Pine Rd,2100.25,2023-03-10
C004,Daniel,Miller,daniel.m@email.com,555-0104,321 Elm St,650.00,2023-04-05
C005,Eva,Garcia,eva.g@email.com,555-0105,654 Maple Dr,1800.90,2023-05-12`;

              const blob = new Blob([sampleData], { type: 'text/csv' });
              const file = new File([blob], 'sample_customers.csv', { type: 'text/csv' });
              handleFileSelect(file);
            }}
            disabled={loading}
          >
            🛒 Customer Data Sample
          </button>
        </div>
      </div>

      {/* Upload Guidelines */}
      <div className="upload-guidelines">
        <h4>📋 Data Upload Guidelines:</h4>
        <ul>
          <li>Ensure your CSV has column headers in the first row</li>
          <li>Remove any completely empty rows or columns</li>
          <li>For best results, include at least 100 rows of data</li>
          <li>PII fields (emails, names, phones) will be automatically detected</li>
          <li>All data is processed securely and deleted after use</li>
        </ul>
      </div>
    </div>
  );
};

export default FileUpload;
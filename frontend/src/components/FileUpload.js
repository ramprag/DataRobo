import React, { useState, useRef } from 'react';
import './FileUpload.css';
import JSZip from 'jszip';


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
    // Enhanced validation for both single files and ZIP files
    const allowedTypes = ['.csv', '.xlsx', '.xls', '.zip'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!allowedTypes.includes(fileExtension)) {
      alert('Please select a CSV, Excel, or ZIP file (.csv, .xlsx, .xls, .zip)');
      return;
    }

    // Validate file size (max 100MB for ZIP, 50MB for single files)
    const maxSize = fileExtension === '.zip' ? 100 * 1024 * 1024 : 50 * 1024 * 1024;
    if (file.size > maxSize) {
      const maxSizeLabel = fileExtension === '.zip' ? '100MB' : '50MB';
      alert(`File size must be less than ${maxSizeLabel}`);
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

  const isZipFile = selectedFile && selectedFile.name.toLowerCase().endsWith('.zip');

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
              <strong>Single Table:</strong> CSV, Excel (.csv, .xlsx, .xls)<br />
              <strong>Multi-Table:</strong> ZIP file containing multiple CSVs
            </div>
            <div className="upload-limit">
              Maximum: 50MB for single files, 100MB for ZIP files
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls,.zip"
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="file-selected">
          <div className="file-info">
            <div className="file-icon">{isZipFile ? '📦' : '📄'}</div>
            <div className="file-details">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-meta">
                <span>{formatFileSize(selectedFile.size)}</span>
                <span>•</span>
                <span>{isZipFile ? 'Multi-table ZIP' : selectedFile.type || 'Single file'}</span>
              </div>
              {isZipFile && (
                <div style={{ fontSize: '0.75rem', color: '#059669', marginTop: '0.25rem' }}>
                  ✨ Multi-table mode: Relationships will be automatically detected
                </div>
              )}
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
                  {isZipFile ? 'Analyzing relationships...' : 'Uploading...'}
                </>
              ) : (
                `📤 Upload ${isZipFile ? 'Multi-Table ' : ''}Dataset`
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

      {/* Enhanced Sample Data Section */}
      <div className="sample-data-section">
        <h4>Don't have data? Try our sample datasets:</h4>
        <div className="sample-datasets">
          {/* Single table samples */}
          <button
            className="btn btn-outline"
            onClick={() => {
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

          {/* Multi-table sample */}
          <button
            className="btn btn-outline"
            onClick={() => {
              // Create sample multi-table ZIP
              createSampleMultiTableZip().then(file => {
                if (file) {
                  handleFileSelect(file);
                }
              });
            }}
            disabled={loading}
            style={{ backgroundColor: '#f0fdf4', borderColor: '#059669', color: '#059669' }}
          >
            🔗 Multi-Table Sample (ZIP)
          </button>
        </div>
      </div>

      {/* Enhanced Upload Guidelines */}
      <div className="upload-guidelines">
        <h4>📋 Data Upload Guidelines:</h4>
        <ul>
          <li><strong>Single Files:</strong> CSV/Excel with headers in first row</li>
          <li><strong>Multi-Table (ZIP):</strong> Multiple CSV files with related data</li>
          <li>Primary/Foreign key relationships will be auto-detected</li>
          <li>Include at least 100 rows per table for best results</li>
          <li>PII fields (emails, names, phones) auto-detected in all tables</li>
          <li>All data processed securely and deleted after use</li>
        </ul>
      </div>
    </div>
  );
};

// Helper function to create sample multi-table ZIP
async function createSampleMultiTableZip() {
  try {
    const customersData = `customer_id,first_name,last_name,email,registration_date
1,Alice,Johnson,alice.j@email.com,2023-01-15
2,Bob,Williams,bob.w@email.com,2023-02-20
3,Carol,Davis,carol.d@email.com,2023-03-10
4,Daniel,Miller,daniel.m@email.com,2023-04-05
5,Eva,Garcia,eva.g@email.com,2023-05-12`;

    const ordersData = `order_id,customer_id,product_name,amount,order_date
101,1,Laptop Pro,1299.99,2023-06-15
102,2,Smartphone X,899.99,2023-06-20
103,1,Wireless Headphones,199.99,2023-07-10
104,3,Tablet Ultra,649.99,2023-07-15
105,2,Smart Watch,299.99,2023-08-01`;

    const productsData = `product_id,product_name,category,price
1,Laptop Pro,Electronics,1299.99
2,Smartphone X,Electronics,899.99
3,Wireless Headphones,Audio,199.99
4,Tablet Ultra,Electronics,649.99
5,Smart Watch,Wearables,299.99`;

    const zip = new JSZip();
    zip.file('customers.csv', customersData);
    zip.file('orders.csv', ordersData);
    zip.file('products.csv', productsData);

    const blob = await zip.generateAsync({ type: 'blob' });
    const file = new File([blob], 'sample_multi_table.zip', { type: 'application/zip' });

    return file;
  } catch (error) {
    console.error('Error creating sample multi-table ZIP:', error);
    alert('Error creating sample data. Please create your own ZIP file with multiple CSVs.');
    return null;
  }
}

export default FileUpload;
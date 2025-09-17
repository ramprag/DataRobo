import React, { useState, useEffect } from 'react';
import './PrivacyConfig.css';

const PrivacyConfig = ({ dataset, onSubmit, loading }) => {
  const [config, setConfig] = useState({
    mask_emails: true,
    mask_names: true,
    mask_phone_numbers: true,
    mask_addresses: true,
    mask_ssn: true,
    custom_fields: [],
    anonymization_method: 'faker'
  });
  const [numRows, setNumRows] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [customField, setCustomField] = useState('');
  const [dataPreview, setDataPreview] = useState(null);

  useEffect(() => {
    // Load data preview to help user configure privacy settings
    fetchDataPreview();
  }, [dataset]);

  const fetchDataPreview = async () => {
    try {
      const response = await fetch(`/api/datasets/${dataset.id}/preview`);
      if (response.ok) {
        const preview = await response.json();
        setDataPreview(preview);
      }
    } catch (error) {
      console.error('Failed to fetch data preview:', error);
    }
  };

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const addCustomField = () => {
    if (customField && !config.custom_fields.includes(customField)) {
      setConfig(prev => ({
        ...prev,
        custom_fields: [...prev.custom_fields, customField]
      }));
      setCustomField('');
    }
  };

  const removeCustomField = (field) => {
    setConfig(prev => ({
      ...prev,
      custom_fields: prev.custom_fields.filter(f => f !== field)
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const rows = numRows ? parseInt(numRows) : null;
    onSubmit(config, rows);
  };

  const getPrivacyLevel = () => {
    const enabledMasks = [
      config.mask_emails,
      config.mask_names,
      config.mask_phone_numbers,
      config.mask_addresses,
      config.mask_ssn
    ].filter(Boolean).length;

    if (enabledMasks >= 4) return { level: 'High', color: 'green' };
    if (enabledMasks >= 2) return { level: 'Medium', color: 'orange' };
    return { level: 'Low', color: 'red' };
  };

  const privacyLevel = getPrivacyLevel();

  const detectPotentialPII = (columns) => {
    const piiSuggestions = [];

    columns.forEach(col => {
      const colLower = col.toLowerCase();

      if (colLower.includes('email') || colLower.includes('mail')) {
        piiSuggestions.push({ column: col, type: 'Email', suggested: config.mask_emails });
      }
      if (colLower.includes('name') || colLower.includes('firstname') || colLower.includes('lastname')) {
        piiSuggestions.push({ column: col, type: 'Name', suggested: config.mask_names });
      }
      if (colLower.includes('phone') || colLower.includes('mobile') || colLower.includes('telephone')) {
        piiSuggestions.push({ column: col, type: 'Phone', suggested: config.mask_phone_numbers });
      }
      if (colLower.includes('address') || colLower.includes('street') || colLower.includes('city')) {
        piiSuggestions.push({ column: col, type: 'Address', suggested: config.mask_addresses });
      }
      if (colLower.includes('ssn') || colLower.includes('social')) {
        piiSuggestions.push({ column: col, type: 'SSN', suggested: config.mask_ssn });
      }
    });

    return piiSuggestions;
  };

  return (
    <form className="privacy-config" onSubmit={handleSubmit}>

      {/* Privacy Level Indicator */}
      <div className="privacy-level">
        <h4>🔒 Privacy Protection Level</h4>
        <div className={`privacy-badge ${privacyLevel.color}`}>
          {privacyLevel.level}
        </div>
        <p>Configure which types of sensitive data to mask or anonymize</p>
      </div>

      {/* Basic Privacy Options */}
      <div className="config-section">
        <h4>🛡️ Standard Privacy Masks</h4>

        <div className="config-options">
          <label className="config-option">
            <input
              type="checkbox"
              checked={config.mask_emails}
              onChange={(e) => handleConfigChange('mask_emails', e.target.checked)}
            />
            <div className="option-content">
              <span className="option-icon">📧</span>
              <div className="option-text">
                <strong>Email Addresses</strong>
                <small>Mask email addresses (e.g., john@example.com → jane@example.org)</small>
              </div>
            </div>
          </label>

          <label className="config-option">
            <input
              type="checkbox"
              checked={config.mask_names}
              onChange={(e) => handleConfigChange('mask_names', e.target.checked)}
            />
            <div className="option-content">
              <span className="option-icon">👤</span>
              <div className="option-text">
                <strong>Names</strong>
                <small>Replace names with fake but realistic names</small>
              </div>
            </div>
          </label>

          <label className="config-option">
            <input
              type="checkbox"
              checked={config.mask_phone_numbers}
              onChange={(e) => handleConfigChange('mask_phone_numbers', e.target.checked)}
            />
            <div className="option-content">
              <span className="option-icon">📱</span>
              <div className="option-text">
                <strong>Phone Numbers</strong>
                <small>Generate fake phone numbers with consistent format</small>
              </div>
            </div>
          </label>

          <label className="config-option">
            <input
              type="checkbox"
              checked={config.mask_addresses}
              onChange={(e) => handleConfigChange('mask_addresses', e.target.checked)}
            />
            <div className="option-content">
              <span className="option-icon">🏠</span>
              <div className="option-text">
                <strong>Addresses</strong>
                <small>Replace with fake but realistic addresses</small>
              </div>
            </div>
          </label>

          <label className="config-option">
            <input
              type="checkbox"
              checked={config.mask_ssn}
              onChange={(e) => handleConfigChange('mask_ssn', e.target.checked)}
            />
            <div className="option-content">
              <span className="option-icon">🆔</span>
              <div className="option-text">
                <strong>Social Security Numbers</strong>
                <small>Replace SSN with fake numbers following valid format</small>
              </div>
            </div>
          </label>
        </div>
      </div>

      {/* PII Detection Results */}
      {dataPreview && (
        <div className="config-section">
          <h4>🔍 Detected Potential PII Fields</h4>
          <div className="pii-detection">
            {detectPotentialPII(dataPreview.columns).length > 0 ? (
              <div className="pii-suggestions">
                {detectPotentialPII(dataPreview.columns).map((suggestion, index) => (
                  <div key={index} className={`pii-suggestion ${suggestion.suggested ? 'protected' : 'unprotected'}`}>
                    <span className="column-name">{suggestion.column}</span>
                    <span className="pii-type">{suggestion.type}</span>
                    <span className={`protection-status ${suggestion.suggested ? 'protected' : 'unprotected'}`}>
                      {suggestion.suggested ? '✅ Protected' : '⚠️ Unprotected'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-pii">No obvious PII fields detected. You may want to review your data manually.</p>
            )}
          </div>
        </div>
      )}

      {/* Advanced Options */}
      <div className="config-section">
        <button
          type="button"
          className="btn btn-text"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          ⚙️ {showAdvanced ? 'Hide' : 'Show'} Advanced Options
        </button>

        {showAdvanced && (
          <div className="advanced-options">

            {/* Anonymization Method */}
            <div className="form-group">
              <label htmlFor="anonymization-method">🔧 Anonymization Method</label>
              <select
                id="anonymization-method"
                value={config.anonymization_method}
                onChange={(e) => handleConfigChange('anonymization_method', e.target.value)}
                className="form-control"
              >
                <option value="faker">Faker (Realistic fake data)</option>
                <option value="hash">Hash (One-way encryption)</option>
                <option value="redact">Redact (Partial masking with *)</option>
              </select>
              <small className="form-help">
                {config.anonymization_method === 'faker' && 'Generates realistic fake data that maintains data utility'}
                {config.anonymization_method === 'hash' && 'One-way hashing - same inputs produce same outputs'}
                {config.anonymization_method === 'redact' && 'Partial redaction with asterisks (e.g., j***@example.com)'}
              </small>
            </div>

            {/* Custom Fields */}
            <div className="form-group">
              <label>📝 Custom Fields to Mask</label>
              <div className="custom-fields">
                <div className="custom-field-input">
                  <input
                    type="text"
                    value={customField}
                    onChange={(e) => setCustomField(e.target.value)}
                    placeholder="Enter column name"
                    className="form-control"
                    list="available-columns"
                  />
                  {dataPreview && (
                    <datalist id="available-columns">
                      {dataPreview.columns.map(col => (
                        <option key={col} value={col} />
                      ))}
                    </datalist>
                  )}
                  <button
                    type="button"
                    onClick={addCustomField}
                    className="btn btn-secondary"
                    disabled={!customField}
                  >
                    Add
                  </button>
                </div>

                {config.custom_fields.length > 0 && (
                  <div className="custom-field-list">
                    {config.custom_fields.map(field => (
                      <div key={field} className="custom-field-tag">
                        <span>{field}</span>
                        <button
                          type="button"
                          onClick={() => removeCustomField(field)}
                          className="remove-field"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Number of Rows */}
            <div className="form-group">
              <label htmlFor="num-rows">📊 Number of Synthetic Rows</label>
              <input
                id="num-rows"
                type="number"
                value={numRows}
                onChange={(e) => setNumRows(e.target.value)}
                placeholder={`Default: ${dataset.row_count?.toLocaleString() || 'Same as original'}`}
                min="1"
                max="1000000"
                className="form-control"
              />
              <small className="form-help">
                Leave empty to generate the same number of rows as the original dataset
              </small>
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <div className="form-actions">
        <button
          type="submit"
          className="btn btn-primary btn-large"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Generating...
            </>
          ) : (
            '🚀 Generate Synthetic Data'
          )}
        </button>

        <div className="generation-info">
          <p>
            📈 Will generate <strong>
              {numRows || dataset.row_count?.toLocaleString() || 'unknown'}
            </strong> rows of synthetic data
          </p>
          <p>
            🔒 Privacy level: <strong className={privacyLevel.color}>
              {privacyLevel.level}
            </strong>
          </p>
        </div>
      </div>

      {/* Privacy Notice */}
      <div className="privacy-notice">
        <h4>🔐 Privacy & Security Notice</h4>
        <ul>
          <li>All data processing happens securely on our servers</li>
          <li>Original data is automatically deleted after synthetic data generation</li>
          <li>Synthetic data maintains statistical properties while protecting individual privacy</li>
          <li>No personally identifiable information is retained</li>
        </ul>
      </div>
    </form>
  );
};

export default PrivacyConfig;
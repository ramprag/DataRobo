// frontend/src/components/AIFabricate.js

import React, { useState } from 'react';
import './AIFabricate.css';

const AIFabricate = ({ onGenerate, loading }) => {
  const [prompt, setPrompt] = useState('');
  const [numRows, setNumRows] = useState(100);
  const [outputFormat, setOutputFormat] = useState('csv');
  const [dataLocale, setDataLocale] = useState('us');

  const examplePrompts = [
    {
      category: 'Healthcare',
      icon: '🏥',
      text: 'Generate 1000 patient records with medical histories, prescriptions, appointment schedules, and insurance information. Include realistic diagnoses and treatment plans.'
    },
    {
      category: 'E-commerce',
      icon: '🛒',
      text: 'Create a dataset of 500 e-commerce customers with names, emails, shipping addresses, order history, and payment preferences. Include realistic purchasing patterns for electronics and household items.'
    },
    {
      category: 'Finance',
      icon: '💰',
      text: 'Generate financial transaction data with account numbers, transaction types, amounts, timestamps, and merchant information. Include both legitimate transactions and fraud indicators.'
    },
    {
      category: 'Social Media',
      icon: '📱',
      text: 'Build a social media dataset with user profiles, posts, comments, likes, shares, and engagement metrics. Include follower relationships and content categories.'
    },
    {
      category: 'HR & Recruitment',
      icon: '👔',
      text: 'Create employee records with personal information, job titles, salaries, performance reviews, attendance, and benefits enrollment. Include department hierarchies.'
    },
    {
      category: 'Education',
      icon: '🎓',
      text: 'Generate student records with enrollment data, course selections, grades, attendance, and extracurricular activities. Include teacher assignments and class schedules.'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim()) {
      onGenerate(prompt, {
        numRows,
        outputFormat,
        dataLocale
      });
    }
  };

  const handleExampleClick = (exampleText) => {
    setPrompt(exampleText);
    // Scroll to textarea
    document.getElementById('ai-prompt-textarea')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  return (
    <div className="ai-fabricate">

      {/* Hero Section */}
      <div className="ai-hero">
        <div className="ai-hero-content">
          <div className="ai-badge">
            <span className="ai-badge-icon">✨</span>
            <span>AI-Powered Generation</span>
          </div>
          <h3>Describe Your Data, We'll Generate It</h3>
          <p>
            No schemas, no configurations—just tell us what data you need in plain English.
            Our AI will understand your requirements and generate realistic, production-ready synthetic data.
          </p>
        </div>
        <div className="ai-hero-stats">
          <div className="stat-card">
            <div className="stat-value">10+</div>
            <div className="stat-label">Data Types</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">50K+</div>
            <div className="stat-label">Records/Gen</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">98%</div>
            <div className="stat-label">Accuracy</div>
          </div>
        </div>
      </div>

      {/* Main Form */}
      <form onSubmit={handleSubmit} className="ai-form">

        {/* Prompt Input */}
        <div className="form-section">
          <label htmlFor="ai-prompt-textarea" className="form-label">
            <span className="label-icon">💬</span>
            Describe Your Data Requirements
          </label>
          <textarea
            id="ai-prompt-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Example: Generate 500 customer records for an online retail store. Include customer names, email addresses, phone numbers, shipping addresses, order history with product names and prices, payment methods, and customer loyalty status. Make sure addresses are realistic US addresses and email formats are valid."
            className="ai-prompt-textarea"
            rows="8"
          />
          <div className="textarea-footer">
            <div className="prompt-tips">
              💡 <strong>Pro tip:</strong> Be specific about data types, relationships, formats, and any business rules
            </div>
            <div className="char-count">
              {prompt.length} characters
            </div>
          </div>
        </div>

        {/* Configuration Options */}
        <div className="config-grid">
          <div className="config-item">
            <label htmlFor="num-rows" className="config-label">
              📊 Number of Records
            </label>
            <input
              id="num-rows"
              type="number"
              value={numRows}
              onChange={(e) => setNumRows(parseInt(e.target.value) || 100)}
              min="1"
              max="50000"
              className="config-input"
            />
            <small className="config-help">Min: 1 | Max: 50,000</small>
          </div>

          <div className="config-item">
            <label htmlFor="output-format" className="config-label">
              📄 Output Format
            </label>
            <select
              id="output-format"
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value)}
              className="config-input"
            >
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
              <option value="parquet">Parquet</option>
              <option value="excel">Excel</option>
            </select>
          </div>

          <div className="config-item">
            <label htmlFor="data-locale" className="config-label">
              🌍 Data Locale
            </label>
            <select
              id="data-locale"
              value={dataLocale}
              onChange={(e) => setDataLocale(e.target.value)}
              className="config-input"
            >
              <option value="us">United States</option>
              <option value="uk">United Kingdom</option>
              <option value="india">India</option>
              <option value="canada">Canada</option>
              <option value="australia">Australia</option>
              <option value="global">Global Mix</option>
            </select>
          </div>
        </div>

        {/* Future Development Notice */}
        <div className="future-notice">
          <div className="notice-content">
            <span className="notice-icon">🚀</span>
            <div>
              <h4>Feature in Development</h4>
              <p>
                AI Fabricate is currently under active development. Our team is integrating
                advanced language models to provide natural language data generation.
                This feature will be available in the next release!
              </p>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !prompt.trim()}
          className="ai-submit-btn"
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              <span>Generating with AI...</span>
            </>
          ) : (
            <>
              <span className="btn-icon">✨</span>
              <span>Fabricate with AI</span>
            </>
          )}
        </button>
      </form>

      {/* Example Prompts */}
      <div className="examples-section">
        <h4 className="examples-title">
          <span className="title-icon">💡</span>
          Example Prompts to Get Started
        </h4>
        <div className="examples-grid">
          {examplePrompts.map((example, idx) => (
            <button
              key={idx}
              onClick={() => handleExampleClick(example.text)}
              className="example-card"
              type="button"
            >
              <div className="example-header">
                <span className="example-icon">{example.icon}</span>
                <span className="example-category">{example.category}</span>
              </div>
              <p className="example-text">{example.text}</p>
              <div className="example-action">
                Use this prompt →
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* How It Works */}
      <div className="how-it-works">
        <h4 className="section-title">How AI Fabricate Works</h4>
        <div className="steps-grid">
          <div className="step-item">
            <div className="step-number">1</div>
            <div className="step-content">
              <h5>Describe Your Needs</h5>
              <p>Write a natural language description of the data you need</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step-item">
            <div className="step-number">2</div>
            <div className="step-content">
              <h5>AI Analysis</h5>
              <p>Our AI understands your requirements and designs the schema</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step-item">
            <div className="step-number">3</div>
            <div className="step-content">
              <h5>Data Generation</h5>
              <p>Synthetic data is generated matching your specifications</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step-item">
            <div className="step-number">4</div>
            <div className="step-content">
              <h5>Download & Use</h5>
              <p>Get production-ready data in your preferred format</p>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="features-section">
        <h4 className="section-title">Why Use AI Fabricate?</h4>
        <div className="features-grid">
          <div className="feature-item">
            <span className="feature-icon">⚡</span>
            <h5>Lightning Fast</h5>
            <p>Generate thousands of records in seconds</p>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🎯</span>
            <h5>Context Aware</h5>
            <p>Understands relationships and business logic</p>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🔒</span>
            <h5>Privacy First</h5>
            <p>All data is synthetic and privacy-safe</p>
          </div>
          <div className="feature-item">
            <span className="feature-icon">🌐</span>
            <h5>Multi-Language</h5>
            <p>Supports multiple locales and formats</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIFabricate;
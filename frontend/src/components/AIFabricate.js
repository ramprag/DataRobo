// frontend/src/components/AIFabricate.js
import React, { useState } from 'react';
import './AIFabricate.css';

const AIFabricate = ({ onGenerate, loading }) => {
  const [prompt, setPrompt] = useState('');
  const [numRows, setNumRows] = useState(100);
  const [outputFormat, setOutputFormat] = useState('csv');
  const [dataLocale, setDataLocale] = useState('us');

  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  const generateSampleData = (category, count, locale) => {
    const rows = [];
    const safeCount = clamp(Number(count || 100) || 100, 100, 200);
    const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
    const names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Riley', 'Casey', 'Jamie', 'Avery', 'Cameron', 'Drew'];
    const citiesUS = ['New York', 'San Francisco', 'Austin', 'Chicago', 'Seattle', 'Denver'];
    const merchants = ['Amazon', 'Target', 'Walmart', 'BestBuy', 'Apple Store', 'Home Depot'];
    const products = ['Laptop', 'Headphones', 'Vacuum', 'Shoes', 'Phone Case', 'Monitor', 'Keyboard'];
    const diagnoses = ['Hypertension', 'Diabetes', 'Asthma', 'Allergy', 'Flu', 'Back Pain'];
    const treatments = ['ACE inhibitor', 'Insulin', 'Inhaler', 'Antihistamine', 'Rest', 'Physical Therapy'];
    for (let i = 0; i < safeCount; i++) {
      const base = { id: i + 1, locale };
      if (category === 'Healthcare') {
        rows.push({
          ...base,
          patient_id: `P-${100000 + i}`,
          name: `${pick(names)} ${String.fromCharCode(65 + (i % 26))}`,
          age: rand(1, 95),
          diagnosis: pick(diagnoses),
          treatment: pick(treatments),
          appointment_date: new Date(Date.now() - rand(0, 365) * 86400000).toISOString().slice(0, 10),
          insurance: pick(['Aetna', 'United', 'Kaiser', 'BlueCross']),
        });
      } else if (category === 'E-commerce') {
        rows.push({
          ...base,
          customer_id: `CUST-${1000 + i}`,
          name: `${pick(names)} ${String.fromCharCode(65 + (i % 26))}`,
          email: `user${1000 + i}@example.com`,
          city: pick(citiesUS),
          product: pick(products),
          quantity: rand(1, 4),
          price: Number((Math.random() * 300 + 10).toFixed(2)),
          order_date: new Date(Date.now() - rand(0, 180) * 86400000).toISOString(),
          payment_method: pick(['card', 'upi', 'cod', 'paypal']),
        });
      } else if (category === 'Finance') {
        rows.push({
          ...base,
          account_id: `ACCT-${5000 + i}`,
          transaction_id: `TX-${Date.now()}-${i}`,
          type: pick(['debit', 'credit']),
          amount: Number((Math.random() * 2000 - 200).toFixed(2)),
          merchant: pick(merchants),
          timestamp: new Date(Date.now() - rand(0, 60) * 86400000).toISOString(),
          fraud_flag: Math.random() < 0.05,
        });
      } else if (category === 'Social Media') {
        rows.push({
          ...base,
          user: `user_${i + 1}`,
          followers: rand(0, 50000),
          posts: rand(0, 2000),
          likes: rand(0, 100000),
          engagement_rate: Number((Math.random() * 10).toFixed(2)),
          category: pick(['tech', 'lifestyle', 'sports', 'finance', 'education']),
        });
      } else if (category === 'HR & Recruitment') {
        rows.push({
          ...base,
          employee_id: `EMP-${100 + i}`,
          name: `${pick(names)} ${String.fromCharCode(65 + (i % 26))}`,
          department: pick(['Engineering', 'Sales', 'HR', 'Finance', 'Ops']),
          title: pick(['Engineer', 'Manager', 'Analyst', 'Specialist']),
          salary: rand(45000, 160000),
          performance_score: rand(1, 5),
        });
      } else if (category === 'Education') {
        rows.push({
          ...base,
          student_id: `S-${10000 + i}`,
          name: `${pick(names)} ${String.fromCharCode(65 + (i % 26))}`,
          grade_level: pick(['Freshman', 'Sophomore', 'Junior', 'Senior']),
          gpa: Number((Math.random() * 2 + 2).toFixed(2)),
          attendance_pct: Number((Math.random() * 20 + 80).toFixed(2)),
        });
      } else {
        rows.push({ ...base, value: i });
      }
    }
    return rows;
  };

  const triggerDownload = (data, format, filenameBase = 'ai_fabricate_sample') => {
    const fmt = (format || 'csv').toLowerCase();
    let blob;
    let filename;
    if (fmt === 'json') {
      const text = JSON.stringify(data, null, 2);
      blob = new Blob([text], { type: 'application/json' });
      filename = `${filenameBase}.json`;
    } else {
      const headers = Array.from(
        data.reduce((set, row) => {
          Object.keys(row).forEach(k => set.add(k));
          return set;
        }, new Set())
      );
      const csvRows = [headers.join(',')].concat(
        data.map(row => headers.map(h => {
          const v = row[h];
          if (v === null || v === undefined) return '';
          const s = String(v).replace(/\"/g, '\"\"');
          return /[\",\n]/.test(s) ? `\"${s}\"` : s;
        }).join(','))
      );
      blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      filename = `${filenameBase}.csv`;
    }
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const examplePrompts = [
    { category: 'Healthcare', icon: '🏥', text: 'Generate 1000 patient records with medical histories, prescriptions, appointment schedules, and insurance information. Include realistic diagnoses and treatment plans.' },
    { category: 'E-commerce', icon: '🛒', text: 'Create a dataset of 500 e-commerce customers with names, emails, shipping addresses, order history, and payment preferences. Include realistic purchasing patterns for electronics and household items.' },
    { category: 'Finance', icon: '💰', text: 'Generate financial transaction data with account numbers, transaction types, amounts, timestamps, and merchant information. Include both legitimate transactions and fraud indicators.' },
    { category: 'Social Media', icon: '📱', text: 'Build a social media dataset with user profiles, posts, comments, likes, shares, and engagement metrics. Include follower relationships and content categories.' },
    { category: 'HR & Recruitment', icon: '👔', text: 'Create employee records with personal information, job titles, salaries, performance reviews, attendance, and benefits enrollment. Include department hierarchies.' },
    { category: 'Education', icon: '🎓', text: 'Generate student records with enrollment data, course selections, grades, attendance, and extracurricular activities. Include teacher assignments and class schedules.' }
  ];

  const detectCategory = (userPrompt) => {
    const match = examplePrompts.find(ep =>
      ep.text === userPrompt ||
      userPrompt.toLowerCase().includes(ep.category.toLowerCase())
    );
    return match?.category || 'Generic';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    const category = detectCategory(prompt);
    const rows = generateSampleData(category, numRows, dataLocale);
    triggerDownload(rows, outputFormat);
    // Preserve existing flow (parent may show a beta alert)
    onGenerate && onGenerate(prompt, { numRows, outputFormat, dataLocale });
  };

  const handleExampleClick = (exampleText) => {
    // Only set the prompt; do NOT generate here (generation happens on submit)
    setPrompt(exampleText);
    const el = document.getElementById('ai-prompt-textarea');
    el && el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  return (
    <div className="ai-fabricate">

      {/* Hero Section */}
      <div className="ai-hero">
        <div className="ai-hero-content">
          <div className="ai-badge">
            <span className="ai-badge-icon">✨</span>
            <span>AI-Powered Generation</span>
            <span className="ai-badge beta" style={{ marginLeft: '8px', background: '#fde68a', color: '#92400e', border: '1px solid #f59e0b' }}>BETA</span>
          </div>
          <h3>Describe Your Data, We'll Generate It</h3>
          <p>
            No schemas, no configurations—just tell us what data you need in plain English.
            Our AI will understand your requirements and generate realistic, production-ready synthetic data.
          </p>
          <p style={{ marginTop: '0.5rem', color: '#92400e', background: '#fef3c7', padding: '8px 12px', borderRadius: '6px', border: '1px solid #f59e0b' }}>
            This is a beta feature. Use example prompts to prefill the description, then click “Fabricate with AI” to download a small sample (100–200 records).
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
            placeholder="Example: Generate 500 customer records for an online retail store..."
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

        {/* Beta Notice */}
        <div className="future-notice">
          <div className="notice-content">
            <span className="notice-icon">🚧</span>
            <div>
              <h4>Beta Notice</h4>
              <p>
                AI Fabricate is in beta. Custom prompts are not yet connected to the backend. Use example prompts to prefill and submit to download a sample dataset instantly.
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
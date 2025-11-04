// frontend/src/components/LandingPage.js
import React from 'react';
import '../styles/Landing.css';

const LandingPage = ({ onGetStarted }) => {
  return (
    <div className="landing-container">
      <nav className="landing-nav">
        <div className="landing-logo">🔒 GSTAN</div>
        <div className="landing-nav-buttons">
          <button onClick={() => onGetStarted('login')} className="landing-btn-secondary">
            Login
          </button>
          <button onClick={() => onGetStarted('signup')} className="landing-btn-primary">
            Sign Up Free
          </button>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="landing-content-wrapper">
          <div className="landing-badge">✨ Enterprise-Grade Synthetic Data Platform</div>
          <h1 className="landing-hero-title">
            Generate Privacy-Safe<br/>Synthetic Data at Scale
          </h1>
          <p className="landing-hero-subtitle">
            Power your AI models with high-quality synthetic data. Preserve privacy, maintain
            statistical accuracy, and accelerate development across healthcare, finance, and beyond.
          </p>
          <div className="landing-hero-buttons">
            <button onClick={() => onGetStarted('signup')} className="landing-btn-cta">
              🚀 Start Creating Data
            </button>
            <button className="landing-btn-demo">📊 View Demo</button>
          </div>
          <div className="landing-stats">
            {[
              ['99.8%', 'Statistical Accuracy'],
              ['10x', 'Faster Development'],
              ['100%', 'Privacy Compliant']
            ].map(([val, label]) => (
              <div key={label} className="landing-stat-item">
                <div className="landing-stat-value">{val}</div>
                <div className="landing-stat-label">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-dark">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Why Synthetic Data Matters</h2>
            <p>The future of AI development depends on high-quality, privacy-preserving data</p>
          </div>
          <div className="landing-features-grid">
            {[
              {
                icon: '🏥',
                title: 'Healthcare Innovation',
                desc: '$150B+ market opportunity. Train models on patient data without privacy risks. 85% faster clinical trial designs.',
                stat: '85%'
              },
              {
                icon: '💰',
                title: 'Financial Services',
                desc: 'Comply with GDPR & PCI-DSS. Test fraud detection models safely. 70% reduction in data breach risks.',
                stat: '70%'
              },
              {
                icon: '🤖',
                title: 'AI Model Training',
                desc: 'Overcome data scarcity. Balance datasets automatically. Achieve 40% better model performance.',
                stat: '40%'
              },
              {
                icon: '🔒',
                title: 'Privacy by Design',
                desc: 'Zero PII exposure. Enterprise-grade anonymization. Maintain regulatory compliance effortlessly.',
                stat: '100%'
              }
            ].map((item) => (
              <div key={item.title} className="landing-feature-card">
                <div className="landing-feature-icon">{item.icon}</div>
                <div className="landing-feature-title">{item.title}</div>
                <p className="landing-feature-desc">{item.desc}</p>
                <div className="landing-feature-stat">{item.stat}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-accent">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Industry Impact</h2>
            <p>Synthetic data is transforming how organizations build AI responsibly</p>
          </div>
          <div className="landing-impact-grid">
            {[
              ['$11.2B', 'Synthetic Data Market by 2030', '📈'],
              ['60%', 'Of AI training data will be synthetic by 2024', '🎯'],
              ['90%', 'Reduction in data collection costs', '💵'],
              ['3x', 'Faster time to market for AI products', '⚡']
            ].map(([stat, desc, icon]) => (
              <div key={desc} className="landing-impact-card">
                <div className="landing-impact-icon">{icon}</div>
                <div className="landing-impact-stat">{stat}</div>
                <div className="landing-impact-desc">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-dark">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Powerful Features</h2>
            <p>Everything you need for enterprise-grade synthetic data generation</p>
          </div>
          <div className="landing-features-list">
            {[
              '🔗 Multi-table relationship detection',
              '🎭 Advanced GAN-based generation',
              '🛡️ HIPAA & GDPR compliant',
              '📊 Real-time quality validation',
              '🔄 CI/CD pipeline integration',
              '☁️ Cloud-native architecture'
            ].map((feature) => (
              <div key={feature} className="landing-feature-item">
                {feature}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-cta">
        <div className="landing-cta-wrapper">
          <h2 className="landing-cta-title">Ready to Transform Your AI Development?</h2>
          <p className="landing-cta-subtitle">
            Join leading enterprises using GSTAN to build AI responsibly and efficiently
          </p>
          <button onClick={() => onGetStarted('signup')} className="landing-btn-cta-large">
            🚀 Get Started Free
          </button>
          <div className="landing-cta-note">
            No credit card required • 14-day free trial • Cancel anytime
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div>© 2024 GSTAN. Built with privacy and security in mind.</div>
      </footer>
    </div>
  );
};

export default LandingPage;
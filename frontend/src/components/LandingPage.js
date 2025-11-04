// frontend/src/components/LandingPage.js
import React, { useEffect, useRef } from 'react';
import '../styles/Landing.css';

const LandingPage = ({ onGetStarted }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 80;

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.radius = Math.random() * 2 + 1;
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
      }

      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(147, 51, 234, 0.6)';
        ctx.fill();
      }
    }

    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, i) => {
        particle.update();
        particle.draw();

        particles.slice(i + 1).forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 120) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.strokeStyle = `rgba(147, 51, 234, ${0.2 * (1 - distance / 120)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      });

      requestAnimationFrame(animate);
    }

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="landing-container">
      <canvas ref={canvasRef} className="landing-canvas" />

      <nav className="landing-nav">
        <div className="landing-logo">
          <span className="logo-icon">🔒</span>
          <span className="logo-text">GSTAN</span>
        </div>
        <div className="landing-nav-buttons">
          <button onClick={() => onGetStarted('login')} className="landing-btn-secondary">
            Sign In
          </button>
          <button onClick={() => onGetStarted('signup')} className="landing-btn-primary">
            Get Started Free
          </button>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="landing-content-wrapper">
          <div className="landing-badge">
            <span className="badge-glow"></span>
            ✨ Enterprise-Grade AI Data Platform
          </div>
          <h1 className="landing-hero-title">
            Generate Privacy-Safe
            <br />
            <span className="gradient-text">Synthetic Data</span>
            <br />
            at Scale
          </h1>
          <p className="landing-hero-subtitle">
            Power your AI models with high-quality synthetic data. Preserve privacy, maintain
            statistical accuracy, and accelerate development with GAN-powered generation.
          </p>
          <div className="landing-hero-buttons">
            <button onClick={() => onGetStarted('signup')} className="landing-btn-cta">
              <span className="btn-icon">🚀</span>
              <span>Start Creating Data</span>
              <span className="btn-shimmer"></span>
            </button>
            <button className="landing-btn-demo">
              <span className="btn-icon">▶️</span>
              <span>Watch Demo</span>
            </button>
          </div>
          <div className="landing-stats">
            <div className="landing-stat-item">
              <div className="landing-stat-value">99.8%</div>
              <div className="landing-stat-label">Statistical Accuracy</div>
            </div>
            <div className="landing-stat-item">
              <div className="landing-stat-value">10x</div>
              <div className="landing-stat-label">Faster Development</div>
            </div>
            <div className="landing-stat-item">
              <div className="landing-stat-value">100%</div>
              <div className="landing-stat-label">Privacy Compliant</div>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-features">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Why Leading Teams Choose GSTAN</h2>
            <p>Transform your data strategy with AI-powered synthetic generation</p>
          </div>
          <div className="landing-features-grid">
            <div className="landing-feature-card">
              <div className="feature-card-glow"></div>
              <div className="landing-feature-icon">🏥</div>
              <div className="landing-feature-title">Healthcare Innovation</div>
              <p className="landing-feature-desc">
                Train models on patient data without privacy risks. 85% faster clinical trial designs with full HIPAA compliance.
              </p>
              <div className="landing-feature-stat">85%</div>
            </div>

            <div className="landing-feature-card">
              <div className="feature-card-glow"></div>
              <div className="landing-feature-icon">💰</div>
              <div className="landing-feature-title">Financial Services</div>
              <p className="landing-feature-desc">
                GDPR & PCI-DSS compliant. Test fraud detection models safely with 70% reduction in data breach risks.
              </p>
              <div className="landing-feature-stat">70%</div>
            </div>

            <div className="landing-feature-card">
              <div className="feature-card-glow"></div>
              <div className="landing-feature-icon">🤖</div>
              <div className="landing-feature-title">AI Model Training</div>
              <p className="landing-feature-desc">
                Overcome data scarcity. Balance datasets automatically. Achieve 40% better model performance with GANs.
              </p>
              <div className="landing-feature-stat">40%</div>
            </div>

            <div className="landing-feature-card">
              <div className="feature-card-glow"></div>
              <div className="landing-feature-icon">🔒</div>
              <div className="landing-feature-title">Privacy by Design</div>
              <p className="landing-feature-desc">
                Zero PII exposure. Enterprise-grade anonymization. Maintain regulatory compliance effortlessly.
              </p>
              <div className="landing-feature-stat">100%</div>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-impact">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Industry Impact</h2>
            <p>Synthetic data is transforming AI development globally</p>
          </div>
          <div className="landing-impact-grid">
            <div className="landing-impact-card">
              <div className="landing-impact-icon">📈</div>
              <div className="landing-impact-stat">$11.2B</div>
              <div className="landing-impact-desc">Synthetic Data Market by 2030</div>
            </div>
            <div className="landing-impact-card">
              <div className="landing-impact-icon">🎯</div>
              <div className="landing-impact-stat">60%</div>
              <div className="landing-impact-desc">Of AI training data will be synthetic by 2024</div>
            </div>
            <div className="landing-impact-card">
              <div className="landing-impact-icon">💵</div>
              <div className="landing-impact-stat">90%</div>
              <div className="landing-impact-desc">Reduction in data collection costs</div>
            </div>
            <div className="landing-impact-card">
              <div className="landing-impact-icon">⚡</div>
              <div className="landing-impact-stat">3x</div>
              <div className="landing-impact-desc">Faster time to market for AI products</div>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-capabilities">
        <div className="landing-content-wrapper">
          <div className="landing-section-header">
            <h2>Enterprise-Grade Capabilities</h2>
            <p>Everything you need for production-ready synthetic data</p>
          </div>
          <div className="landing-capabilities-grid">
            <div className="landing-capability-item">
              <span className="capability-icon">🔗</span>
              <span>Multi-table relationship detection</span>
            </div>
            <div className="landing-capability-item">
              <span className="capability-icon">🎭</span>
              <span>Advanced GAN-based generation</span>
            </div>
            <div className="landing-capability-item">
              <span className="capability-icon">🛡️</span>
              <span>HIPAA & GDPR compliant</span>
            </div>
            <div className="landing-capability-item">
              <span className="capability-icon">📊</span>
              <span>Real-time quality validation</span>
            </div>
            <div className="landing-capability-item">
              <span className="capability-icon">🔄</span>
              <span>CI/CD pipeline integration</span>
            </div>
            <div className="landing-capability-item">
              <span className="capability-icon">☁️</span>
              <span>Cloud-native architecture</span>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-cta">
        <div className="landing-cta-wrapper">
          <div className="cta-glow"></div>
          <h2 className="landing-cta-title">Ready to Transform Your AI Development?</h2>
          <p className="landing-cta-subtitle">
            Join leading enterprises using GSTAN to build AI responsibly and efficiently
          </p>
          <button onClick={() => onGetStarted('signup')} className="landing-btn-cta-large">
            <span className="btn-icon">🚀</span>
            <span>Get Started Free</span>
            <span className="btn-shimmer"></span>
          </button>
          <div className="landing-cta-note">
            No credit card required • 14-day free trial • Cancel anytime
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <div className="landing-logo">
              <span className="logo-icon">🔒</span>
              <span className="logo-text">GSTAN</span>
            </div>
            <p>Enterprise synthetic data generation</p>
          </div>
          <div className="footer-copyright">
            © 2024 GSTAN. Built with privacy and security in mind.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
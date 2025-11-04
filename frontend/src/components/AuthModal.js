// frontend/src/components/AuthModal.js
import React, { useState } from 'react';
import '../styles/Auth.css';

const AuthModal = ({ show, mode, onClose, onSuccess }) => {
  const [isLogin, setIsLogin] = useState(mode === 'login');
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  if (!show) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      const users = JSON.parse(localStorage.getItem('gstan_users') || '[]');

      if (isLogin) {
        const user = users.find(
          (u) => u.email === formData.email && u.password === formData.password
        );
        if (!user) {
          setError('Invalid email or password');
          setLoading(false);
          return;
        }
        localStorage.setItem(
          'gstan_auth',
          JSON.stringify({ email: user.email, username: user.username })
        );
        setSuccess('Login successful! Redirecting...');
        setTimeout(() => {
          onSuccess(user);
        }, 1000);
      } else {
        if (users.some((u) => u.email === formData.email)) {
          setError('Email already registered');
          setLoading(false);
          return;
        }
        if (!formData.username || !formData.email || !formData.password) {
          setError('All fields are required');
          setLoading(false);
          return;
        }
        if (formData.password.length < 6) {
          setError('Password must be at least 6 characters');
          setLoading(false);
          return;
        }

        const newUser = {
          username: formData.username,
          email: formData.email,
          password: formData.password,
          createdAt: new Date().toISOString()
        };

        users.push(newUser);
        localStorage.setItem('gstan_users', JSON.stringify(users));
        localStorage.setItem(
          'gstan_auth',
          JSON.stringify({ email: newUser.email, username: newUser.username })
        );
        setSuccess('Account created successfully! Redirecting...');
        setTimeout(() => {
          onSuccess(newUser);
        }, 1000);
      }
    } catch (err) {
      console.error('Auth error:', err);
      setError('Authentication failed. Please try again.');
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccess('');
    setFormData({ username: '', email: '', password: '' });
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="auth-modal-close" onClick={onClose}>
          ×
        </button>

        <div className="auth-modal-header">
          <div className="auth-modal-icon">🔒</div>
          <h2 className="auth-modal-title">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p className="auth-modal-subtitle">
            {isLogin ? 'Sign in to continue' : 'Start generating synthetic data'}
          </p>
        </div>

        {error && (
          <div className="auth-alert auth-alert-error">
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="auth-alert auth-alert-success">
            <span>{success}</span>
          </div>
        )}

        <div className="auth-form">
          {!isLogin && (
            <div className="auth-form-group">
              <label className="auth-label">Username</label>
              <input
                type="text"
                className="auth-input"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                placeholder="johndoe"
                disabled={loading}
              />
            </div>
          )}

          <div className="auth-form-group">
            <label className="auth-label">Email</label>
            <input
              type="email"
              className="auth-input"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="you@company.com"
              disabled={loading}
            />
          </div>

          <div className="auth-form-group">
            <label className="auth-label">Password</label>
            <input
              type="password"
              className="auth-input"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              placeholder="••••••••"
              disabled={loading}
            />
          </div>

          <button
            onClick={handleSubmit}
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? (
              <span>Processing...</span>
            ) : (
              <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
            )}
          </button>
        </div>

        <div className="auth-toggle">
          <button onClick={toggleMode} className="auth-toggle-btn" disabled={loading}>
            {isLogin
              ? "Don't have an account? Sign up"
              : 'Already have an account? Sign in'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;
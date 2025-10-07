// frontend/src/components/GANConfig.js
import React, { useState, useEffect } from 'react';
import './GANConfig.css';

const GANConfig = ({ onConfigChange, initialConfig = {} }) => {
  const [ganAvailable, setGanAvailable] = useState(null);
  const [availableMethods, setAvailableMethods] = useState([]);
  const [config, setConfig] = useState({
    use_gan: initialConfig.use_gan !== undefined ? initialConfig.use_gan : true,
    gan_model: initialConfig.gan_model || 'ctgan'
  });

  useEffect(() => {
    checkGANAvailability();
  }, []);

  useEffect(() => {
    if (onConfigChange) {
      onConfigChange(config);
    }
  }, [config]);

  const checkGANAvailability = async () => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || '';
      const response = await fetch(`${apiUrl}/api/methods`);

      if (response.ok) {
        const data = await response.json();
        setGanAvailable(data.gan_available);
        setAvailableMethods(data.methods);

        // If GAN not available, disable it
        if (!data.gan_available && config.use_gan) {
          setConfig(prev => ({ ...prev, use_gan: false }));
        }
      }
    } catch (error) {
      console.error('Failed to check GAN availability:', error);
      setGanAvailable(false);
    }
  };

  const handleToggleGAN = (enabled) => {
    setConfig(prev => ({ ...prev, use_gan: enabled }));
  };

  const handleModelChange = (model) => {
    setConfig(prev => ({ ...prev, gan_model: model }));
  };

  return (
    <div className="gan-config">
      <div className="config-section">
        <h4>🤖 Generation Method</h4>

        <div className="method-selector">
          <label className="method-option">
            <input
              type="radio"
              name="generation-method"
              checked={config.use_gan}
              onChange={() => handleToggleGAN(true)}
              disabled={!ganAvailable}
            />
            <div className="option-content">
              <div className="option-header">
                <span className="option-icon">🤖</span>
                <strong>GAN Model (Recommended)</strong>
                {ganAvailable === false && (
                  <span className="badge badge-warning">Not Available</span>
                )}
                {ganAvailable === true && (
                  <span className="badge badge-success">Available</span>
                )}
              </div>
              <div className="option-description">
                Uses deep learning (CTGAN/TVAE) for high-quality synthetic data.
                Best for complex datasets with intricate relationships.
              </div>
              {config.use_gan && ganAvailable && (
                <div className="model-selection">
                  <label className="model-radio">
                    <input
                      type="radio"
                      name="gan-model"
                      value="ctgan"
                      checked={config.gan_model === 'ctgan'}
                      onChange={(e) => handleModelChange(e.target.value)}
                    />
                    <span>
                      <strong>CTGAN</strong> - Best for mixed data types (categorical + numerical)
                    </span>
                  </label>

                  <label className="model-radio">
                    <input
                      type="radio"
                      name="gan-model"
                      value="tvae"
                      checked={config.gan_model === 'tvae'}
                      onChange={(e) => handleModelChange(e.target.value)}
                    />
                    <span>
                      <strong>TVAE</strong> - Faster training, good for numerical data
                    </span>
                  </label>
                </div>
              )}
            </div>
          </label>

          <label className="method-option">
            <input
              type="radio"
              name="generation-method"
              checked={!config.use_gan}
              onChange={() => handleToggleGAN(false)}
            />
            <div className="option-content">
              <div className="option-header">
                <span className="option-icon">📊</span>
                <strong>Statistical Method</strong>
                <span className="badge badge-info">Always Available</span>
              </div>
              <div className="option-description">
                Uses statistical distributions to generate data.
                Faster but may not capture complex patterns. Good for simple datasets.
              </div>
            </div>
          </label>
        </div>

        {ganAvailable === false && (
          <div className="info-box info-warning">
            <span className="info-icon">⚠️</span>
            <div>
              <strong>GAN Models Not Available</strong>
              <p>
                GAN libraries are not installed on the server.
                The system will use statistical method instead.
                To enable GAN support, install: <code>pip install sdv torch</code>
              </p>
            </div>
          </div>
        )}

        {config.use_gan && ganAvailable && (
          <div className="info-box info-success">
            <span className="info-icon">✨</span>
            <div>
              <strong>GAN Generation Enabled</strong>
              <p>
                Your synthetic data will be generated using {config.gan_model.toUpperCase()}.
                This may take longer but produces higher quality results.
                {config.gan_model === 'ctgan' && ' CTGAN is optimal for datasets with mixed data types.'}
                {config.gan_model === 'tvae' && ' TVAE trains faster and works well with numerical data.'}
              </p>
            </div>
          </div>
        )}

        <div className="comparison-table">
          <h5>📋 Method Comparison</h5>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>GAN (CTGAN/TVAE)</th>
                <th>Statistical</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Data Quality</td>
                <td className="text-success">⭐⭐⭐⭐⭐ Excellent</td>
                <td className="text-warning">⭐⭐⭐ Good</td>
              </tr>
              <tr>
                <td>Speed</td>
                <td className="text-warning">⚡ Slower (training required)</td>
                <td className="text-success">⚡⚡⚡ Very Fast</td>
              </tr>
              <tr>
                <td>Complex Patterns</td>
                <td className="text-success">✓ Captures well</td>
                <td className="text-warning">△ Limited</td>
              </tr>
              <tr>
                <td>Dataset Size</td>
                <td className="text-info">Best for 100+ rows</td>
                <td className="text-success">Works with any size</td>
              </tr>
              <tr>
                <td>Privacy</td>
                <td className="text-success">✓ High (with masking)</td>
                <td className="text-success">✓ High (with masking)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default GANConfig;
// frontend/src/components/MultimodalStudio.js
import React, { useState } from 'react';
import './MultimodalStudio.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const MultimodalStudio = () => {
  const [name, setName] = useState('Investor Demo');
  // Text
  const [textEnabled, setTextEnabled] = useState(true);
  const [textCount, setTextCount] = useState(1000);
  const [textDomain, setTextDomain] = useState('generic');
  // Image (diffusion ONNX)
  const [imageEnabled, setImageEnabled] = useState(true);
  const [imageCount, setImageCount] = useState(30);
  const [imageWidth, setImageWidth] = useState(256);
  const [imageHeight, setImageHeight] = useState(256);
  const [imageDomain, setImageDomain] = useState('automotive');
  const [promptsText, setPromptsText] = useState('');

  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const exampleSets = [
    {
      title: 'Warehouse Inventory (Text + Images)',
      icon: '📦',
      desc: 'Generate 120 inventory descriptions and 6 placeholder images',
      handler: () => generateStaticSample('warehouse')
    },
    {
      title: 'Autonomous Driving (Text + Images)',
      icon: '🚗',
      desc: 'Generate 100 scene captions and 4 placeholder images',
      handler: () => generateStaticSample('automotive')
    }
  ];

  const generateStaticSample = (domain) => {
    const countText = domain === 'warehouse' ? 120 : 100;
    const countImages = domain === 'warehouse' ? 6 : 4;
    const captions = [];
    for (let i = 0; i < countText; i++) {
      if (domain === 'warehouse') {
        captions.push(`Item ${i + 1}: ${['Pallet', 'Box', 'Crate', 'Bin'][i % 4]} - ${['Fragile', 'Heavy', 'Perishable', 'Standard'][i % 4]} - Aisle ${1 + (i % 12)}`);
      } else {
        captions.push(`Frame ${i + 1}: ${['Car', 'Pedestrian', 'Cyclist'][i % 3]} detected near ${['intersection', 'crosswalk', 'highway exit'][i % 3]}`);
      }
    }
    const images = Array.from({ length: countImages }).map((_, i) => ({
      file: `synthetic_${domain}_${i + 1}.png`,
      width: 256,
      height: 256,
      note: 'Placeholder (beta) — generation mocked client-side'
    }));
    setJob({
      id: `static-${domain}-${Date.now()}`,
      status: 'completed',
      output: {
        text: {
          requested_count: countText,
          domain: domain,
          sample: captions.slice(0, 5)
        },
        image: {
          images_dir: `/static/${domain}`,
          count: countImages,
          width: 256,
          height: 256,
          metadata: `${countImages} placeholder images generated (beta)`,
          files: images
        }
      },
      message: 'Multimodal static sample generated (beta)'
    });
  };

  const start = async () => {
    setLoading(true);
    setError(null);
    try {
      const prompts = promptsText
        .split('\n')
        .map(s => s.trim())
        .filter(Boolean);

      // Beta path: if no prompts provided, generate a static sample instead of calling backend
      if (prompts.length === 0) {
        generateStaticSample(imageDomain || textDomain || 'generic');
        return;
      }

      const body = {
        name,
        spec: {
          text: { enabled: textEnabled, count: Number(textCount), domain: textDomain },
          image: {
            enabled: imageEnabled,
            count: Number(imageCount),
            width: Number(imageWidth),
            height: Number(imageHeight),
            domain: imageDomain,
            prompts,
            num_inference_steps: 15,
            guidance_scale: 6.0
          }
        }
      };

      const resp = await fetch(`${API_BASE_URL}/api/multimodal/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to start job');
      const data = await resp.json();
      setJob(data);
      poll(data.id);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const poll = async (jobId) => {
    let tries = 0;
    const maxTries = 600;
    const interval = setInterval(async () => {
      tries++;
      try {
        const r = await fetch(`${API_BASE_URL}/api/multimodal/jobs/${jobId}`);
        const d = await r.json();
        setJob(d);
        if (d.status === 'completed' || d.status === 'failed') {
          clearInterval(interval);
        }
      } catch {}
      if (tries > maxTries) clearInterval(interval);
    }, 3000);
  };

  return (
    <div className="mm-studio">
      <div className="mm-hero">
        <div className="mm-hero-content">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <h2>Multimodal Synthetic Studio</h2>
            <span style={{ background: '#fde68a', color: '#92400e', border: '1px solid #f59e0b', padding: '2px 8px', borderRadius: '6px', fontWeight: 700 }}>BETA</span>
          </div>
          <p>CPU-friendly diffusion (ONNX) for image generation + your existing text pipelines.</p>
          <p style={{ opacity: 0.9 }}>Best results: 256×256, steps 10–20. Provide prompts or choose a domain.</p>
          <p style={{ marginTop: '0.5rem', color: '#92400e', background: '#fef3c7', padding: '8px 12px', borderRadius: '6px', border: '1px solid #f59e0b' }}>
            This is a beta feature. If no prompts are provided, clicking Start will generate a static sample instantly.
          </p>
        </div>
      </div>

      {error && <div className="mm-alert error"><span>{error}</span></div>}

      <div className="mm-grid">
        <div className="mm-card">
          <h3>Example Presets (Beta)</h3>
          <div style={{ display: 'grid', gap: '8px' }}>
            {exampleSets.map((ex, idx) => (
              <button key={idx} className="mm-btn" onClick={ex.handler}>
                <span style={{ marginRight: '6px' }}>{ex.icon}</span>
                {ex.title}
                <span style={{ marginLeft: '8px', opacity: 0.8 }}>— {ex.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mm-card">
          <h3>Text</h3>
          <label className="mm-switch">
            <input type="checkbox" checked={textEnabled} onChange={(e) => setTextEnabled(e.target.checked)} />
            <span>Enabled</span>
          </label>
          <div className="mm-row">
            <label>Records</label>
            <input type="number" value={textCount} min="1" max="1000000" onChange={(e) => setTextCount(e.target.value)} />
          </div>
          <div className="mm-row">
            <label>Domain</label>
            <select value={textDomain} onChange={(e) => setTextDomain(e.target.value)}>
              <option value="generic">Generic</option>
              <option value="ecommerce">E-commerce</option>
              <option value="automotive">Automotive</option>
              <option value="warehouse">Warehouse</option>
              <option value="robotics">Robotics</option>
            </select>
          </div>
        </div>

        <div className="mm-card">
          <h3>Images (Diffusion ONNX)</h3>
          <label className="mm-switch">
            <input type="checkbox" checked={imageEnabled} onChange={(e) => setImageEnabled(e.target.checked)} />
            <span>Enabled</span>
          </label>
          <div className="mm-row">
            <label>Images</label>
            <input type="number" value={imageCount} min="1" max="200" onChange={(e) => setImageCount(e.target.value)} />
          </div>
          <div className="mm-row two">
            <div>
              <label>Width</label>
              <input type="number" value={imageWidth} min="64" max="512" onChange={(e) => setImageWidth(e.target.value)} />
            </div>
            <div>
              <label>Height</label>
              <input type="number" value={imageHeight} min="64" max="512" onChange={(e) => setImageHeight(e.target.value)} />
            </div>
          </div>
          <div className="mm-row">
            <label>Domain</label>
            <select value={imageDomain} onChange={(e) => setImageDomain(e.target.value)}>
              <option value="automotive">Automotive</option>
              <option value="warehouse">Warehouse</option>
              <option value="robotics">Robotics</option>
              <option value="generic">Generic</option>
            </select>
          </div>
          <div className="mm-row">
            <label>Prompts</label>
          </div>
          <textarea
            className="mm-textarea"
            placeholder="One prompt per line (optional). If empty, domain prompts are used."
            value={promptsText}
            onChange={(e) => setPromptsText(e.target.value)}
            rows={6}
          />
        </div>

        <div className="mm-card">
          <h3>Job</h3>
          <div className="mm-row">
            <label>Name</label>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <button className="mm-btn" disabled={loading} onClick={start}>
            {loading ? 'Starting…' : '🚀 Start Multimodal Generation'}
          </button>

          {job && (
            <div className="mm-job">
              <div className={`mm-status ${job.status}`}>Status: {job.status}</div>
              {job.output && (
                <div className="mm-output">
                  {job.output.text && (
                    <div className="mm-section">
                      <h4>Text</h4>
                      <p>Requested: {job.output.text.requested_count} records</p>
                      <p>Domain: {job.output.text.domain}</p>
                    </div>
                  )}
                  {job.output.image && (
                    <div className="mm-section">
                      <h4>Images</h4>
                      <p>Images Dir: {job.output.image.images_dir}</p>
                      <p>Count: {job.output.image.count}</p>
                      <p>Resolution: {job.output.image.width}×{job.output.image.height}</p>
                      <p>Metadata: {job.output.image.metadata}</p>
                    </div>
                  )}
                </div>
              )}
              {job.message && <div className="mm-message">Message: {job.message}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MultimodalStudio;
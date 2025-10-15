// frontend/src/components/CICDPanel.js
import React, { useEffect, useState } from 'react';

const CICDPanel = ({ api, selectedDataset, onBackHome }) => {
  const [pipelines, setPipelines] = useState([]);
  const [runs, setRuns] = useState([]);
  const [saving, setSaving] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const [form, setForm] = useState({
    id: '',
    name: 'Default Pipeline',
    description: 'Automated synthetic data CI/CD pipeline',
    auto_trigger_on_upload: false,
    default_privacy: {
      mask_emails: true,
      mask_names: true,
      mask_phone_numbers: true,
      mask_addresses: true,
      mask_ssn: true,
      custom_fields: [],
      use_gan: true,
      gan_epochs: 100,
      anonymization_method: 'faker'
    },
    quality_gate: { min_overall_quality: 60, allow_missing_columns: true },
    export_target: { type: 'local', path: 'artifacts' },
    active: true
  });

  const fetchAll = async () => {
    const [pRes, rRes] = await Promise.all([
      api.get('/api/cicd/pipelines'),
      api.get('/api/cicd/runs')
    ]);
    setPipelines(pRes.data || []);
    setRuns(rRes.data || []);
  };

  useEffect(() => {
    fetchAll().catch(console.error);
    const id = setInterval(() => {
      api.get('/api/cicd/runs').then(r => setRuns(r.data || [])).catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, []);

  const savePipeline = async () => {
    try {
      setSaving(true);
      const res = await api.post('/api/cicd/pipelines', form);
      setForm(prev => ({ ...prev, id: res.data.id }));
      await fetchAll();
    } catch (e) {
      console.error(e);
      alert('Failed to save pipeline');
    } finally {
      setSaving(false);
    }
  };

  const triggerRun = async (pipelineId) => {
    if (!selectedDataset) {
      alert('Select or create a dataset first');
      return;
    }
    try {
      setTriggering(true);
      await api.post(`/api/cicd/pipelines/${pipelineId}/run`, { dataset_id: selectedDataset.id });
      await fetchAll();
    } catch (e) {
      console.error(e);
      alert('Failed to start pipeline run');
    } finally {
      setTriggering(false);
    }
  };

  const latestArtifact = (run) => {
    if (!run?.artifact_path) return null;
    const name = run.artifact_path.split('/').pop();
    return { name, path: run.artifact_path };
  };

  return (
    <div className="cicd-panel">
      <div className="cicd-header">
        <div>
          <h2>CI/CD Pipelines</h2>
          <p>Automate synthetic data generation with quality gates and exports</p>
        </div>
        <div className="cicd-actions">
          <button className="btn" onClick={onBackHome}>← Back to Home</button>
          <button className="btn btn-primary" onClick={savePipeline} disabled={saving}>
            {saving ? 'Saving…' : 'Save Pipeline'}
          </button>
        </div>
      </div>

      <div className="cicd-grid">
        <section className="card">
          <h3>Pipeline Configuration</h3>
          <div className="form-grid">
            <label>
              <span>Name</span>
              <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
            </label>
            <label>
              <span>Description</span>
              <input value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} />
            </label>
            <label className="toggle">
              <input type="checkbox" checked={form.auto_trigger_on_upload}
                     onChange={e => setForm({ ...form, auto_trigger_on_upload: e.target.checked })} />
              <span>Auto-trigger on Upload</span>
            </label>
            <div className="subsection">
              <h4>Default Privacy</h4>
              <div className="inline">
                <label className="toggle">
                  <input type="checkbox" checked={form.default_privacy.use_gan}
                         onChange={e => setForm({ ...form, default_privacy: { ...form.default_privacy, use_gan: e.target.checked } })} />
                  <span>Use GAN</span>
                </label>
                <label>
                  <span>GAN Epochs</span>
                  <input type="number" min="10" max="500"
                         value={form.default_privacy.gan_epochs}
                         onChange={e => setForm({ ...form, default_privacy: { ...form.default_privacy, gan_epochs: parseInt(e.target.value || 0, 10) } })} />
                </label>
              </div>
            </div>
            <div className="subsection">
              <h4>Quality Gate</h4>
              <label>
                <span>Min Overall Quality</span>
                <input type="number" min="0" max="100"
                       value={form.quality_gate.min_overall_quality}
                       onChange={e => setForm({ ...form, quality_gate: { ...form.quality_gate, min_overall_quality: parseFloat(e.target.value || 0) } })} />
              </label>
            </div>
            <div className="subsection">
              <h4>Export Target</h4>
              <div className="inline">
                <label>
                  <span>Type</span>
                  <select value={form.export_target.type}
                          onChange={e => setForm({ ...form, export_target: { ...form.export_target, type: e.target.value } })}>
                    <option value="local">Local</option>
                    <option value="s3">S3 (stub)</option>
                    <option value="gcs">GCS (stub)</option>
                    <option value="azure">Azure (stub)</option>
                  </select>
                </label>
                <label>
                  <span>Path/Bucket</span>
                  <input value={form.export_target.path}
                         onChange={e => setForm({ ...form, export_target: { ...form.export_target, path: e.target.value } })} />
                </label>
              </div>
            </div>
          </div>
        </section>

        <section className="card">
          <h3>Available Pipelines</h3>
          <ul className="pipelines-list">
            {pipelines.map(p => (
              <li key={p.id} className="pipeline-item">
                <div>
                  <div className="title">{p.name}</div>
                  <div className="subtitle">{p.description}</div>
                  <div className="meta">
                    <span>{p.active ? 'Active' : 'Inactive'}</span>
                    <span>•</span>
                    <span>{p.auto_trigger_on_upload ? 'Auto-trigger ON' : 'Manual'}</span>
                  </div>
                </div>
                <div className="row-actions">
                  <button className="btn btn-secondary" disabled={!selectedDataset || triggering}
                          onClick={() => triggerRun(p.id)}>
                    {triggering ? 'Starting…' : 'Run on Selected Dataset'}
                  </button>
                </div>
              </li>
            ))}
            {pipelines.length === 0 && <li className="pipeline-item empty">No pipelines yet. Create and save one.</li>}
          </ul>
        </section>

        <section className="card">
          <h3>Recent Runs</h3>
          <ul className="runs-list">
            {runs.map(run => {
              const artifact = latestArtifact(run);
              return (
                <li key={run.id} className={`run-item status-${run.status}`}>
                  <div className="row">
                    <div className="status-pill">{run.status}</div>
                    <div className="grow">
                      <div className="title">Run {run.id.slice(0, 8)} · Pipeline {run.pipeline_id.slice(0, 6)}</div>
                      <div className="subtitle">{run.message}</div>
                      <div className="meta">
                        <span>Progress: {run.progress}%</span>
                        <span>•</span>
                        <span>Started: {new Date(run.started_at).toLocaleString()}</span>
                        {run.finished_at && <>
                          <span>•</span>
                          <span>Finished: {new Date(run.finished_at).toLocaleString()}</span>
                        </>}
                      </div>
                    </div>
                    {artifact && (
                      <div>
                        <a className="btn btn-link" href={`/${artifact.path}`} download>
                          Download {artifact.name}
                        </a>
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
            {runs.length === 0 && <li className="run-item empty">No runs yet.</li>}
          </ul>
        </section>
      </div>
    </div>
  );
};

export default CICDPanel;
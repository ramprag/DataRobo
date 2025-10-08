import React, { useState, useEffect, useCallback } from 'react';
import './RelationshipViewer.css';

const RelationshipViewer = ({ datasetId, relationshipData, datasetStatus }) => {
  const [relationships, setRelationships] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRelationships = useCallback(async () => {
    if (!datasetId) return;

    try {
      setLoading(true);
      setError(null);

      const apiUrl = process.env.REACT_APP_API_URL || '';
      const url = `${apiUrl}/api/datasets/${datasetId}/relationships`;

      console.log('Fetching relationships from:', url);
      const response = await fetch(url);

      if (response.ok) {
        const data = await response.json();
        console.log('Received relationship data:', data);
        setRelationships(data);
      } else {
        throw new Error('Failed to load relationships');
      }
    } catch (err) {
      console.error('Error fetching relationships:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    if (relationshipData) {
      console.log('Using provided relationship data:', relationshipData);
      setRelationships(relationshipData);
    } else if (datasetId) {
      fetchRelationships();
    }
  }, [datasetId, relationshipData, fetchRelationships]);

  // Poll for updates when status is processing or completed
  useEffect(() => {
    if (!datasetId) return;

    let intervalId;

    if (datasetStatus === 'processing') {
      // Poll every 2 seconds while processing
      intervalId = setInterval(() => {
        console.log('Polling for relationship updates...');
        fetchRelationships();
      }, 2000);
    } else if (datasetStatus === 'completed' && !relationships?.relationship_summary?.total_relationships) {
      // Fetch once when completed if no relationships yet
      fetchRelationships();
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [datasetId, datasetStatus, relationships, fetchRelationships]);

  if (loading && !relationships) {
    return (
      <div className="relationship-viewer loading">
        <h4>🔗 Table Relationships</h4>
        <div className="loading-content">
          <span className="spinner"></span>
          <p>Analyzing relationships...</p>
        </div>
      </div>
    );
  }

  if (error && !relationships) {
    return (
      <div className="relationship-viewer error">
        <h4>🔗 Table Relationships</h4>
        <div className="error-content">
          <p>Error loading relationships: {error}</p>
          <button onClick={fetchRelationships} className="btn btn-text">
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  if (!relationships) {
    return (
      <div className="relationship-viewer loading">
        <h4>🔗 Table Relationships</h4>
        <p>Waiting for relationship data...</p>
      </div>
    );
  }

  // Check if single table
  if (relationships.table_count <= 1) {
    return (
      <div className="relationship-viewer single-table">
        <h4>📄 Single Table Dataset</h4>
        <p>This dataset contains a single table. No relationships to analyze.</p>
      </div>
    );
  }

  const summary = relationships.relationship_summary || {};
  const totalRelationships = summary.total_relationships || 0;
  const relationshipDetails = summary.relationship_details || [];

  console.log('Rendering with:', { totalRelationships, relationshipDetails, summary });

  return (
    <div className="relationship-viewer">
      <div className="relationship-header">
        <h4>🔗 Table Relationships Detected</h4>
        <div className="relationship-stats">
          <span className="stat-item">
            <strong>{relationships.table_count}</strong> tables
          </span>
          <span className="stat-item">
            <strong>{totalRelationships}</strong> relationships
          </span>
          <span className="stat-item">
            <strong>{summary.tables_with_primary_keys || 0}</strong> with primary keys
          </span>
        </div>
      </div>

      {totalRelationships > 0 ? (
        <>
          <div className="generation-order">
            <h5>📋 Generation Order</h5>
            <div className="order-flow">
              {(summary.generation_order || []).map((table, index) => (
                <React.Fragment key={table}>
                  <div className="table-node">
                    <span className="table-name">{table}</span>
                    <span className="table-order">{index + 1}</span>
                  </div>
                  {index < (summary.generation_order || []).length - 1 && (
                    <div className="flow-arrow">→</div>
                  )}
                </React.Fragment>
              ))}
            </div>
            <small className="order-explanation">
              Parent tables are generated first to ensure referential integrity
            </small>
          </div>

          <div className="relationships-details">
            <h5>🔍 Relationship Details</h5>
            <div className="relationship-list">
              {relationshipDetails.map((rel, index) => (
                <div key={index} className="relationship-item">
                  <div className="relationship-description">
                    {rel.description}
                  </div>
                  <div className="relationship-confidence">
                    <span className="confidence-label">Confidence:</span>
                    <span className={`confidence-value ${
                      rel.confidence >= 0.9 ? 'high' :
                      rel.confidence >= 0.7 ? 'medium' : 'low'
                    }`}>
                      {Math.round(rel.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="no-relationships">
          <p>⚠️ No relationships detected between tables.</p>
          <p>Each table will be generated independently.</p>
          {process.env.NODE_ENV === 'development' && (
            <details style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#6b7280' }}>
              <summary>Debug Information</summary>
              <pre style={{ textAlign: 'left', fontSize: '0.75rem', overflow: 'auto' }}>
                {JSON.stringify({
                  summary,
                  relationships,
                  datasetStatus
                }, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      <div className="relationship-notice">
        <h6>ℹ️ How Relationship Detection Works</h6>
        <ul>
          <li>Primary keys are identified by uniqueness and naming patterns</li>
          <li>Foreign keys are detected by value overlap with primary keys</li>
          <li>Synthetic data preserves these relationships automatically</li>
          <li>High confidence (>90%) relationships are most reliable</li>
        </ul>
      </div>
    </div>
  );
};

export default RelationshipViewer;
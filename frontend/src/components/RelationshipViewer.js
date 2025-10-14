import React, { useState, useEffect } from 'react';
import './RelationshipViewer.css';

const RelationshipViewer = ({ datasetId, relationshipData }) => {
  const [relationships, setRelationships] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('RelationshipViewer - Props received:', { datasetId, relationshipData });

    if (relationshipData) {
      // Use the relationship data passed directly
      console.log('RelationshipViewer - Using passed relationship data');
      setRelationships({
        dataset_id: datasetId,
        table_count: relationshipData.table_count || 1,
        relationships: relationshipData.relationships || {},
        relationship_summary: relationshipData.relationship_summary || {
          total_relationships: 0,
          tables_with_primary_keys: 0,
          tables_with_foreign_keys: 0,
          generation_order: [],
          relationship_details: []
        },
        status: relationshipData.status || 'unknown'
      });
    } else if (datasetId) {
      // Fetch from API
      console.log('RelationshipViewer - Fetching from API');
      fetchRelationships();
    }
  }, [datasetId, relationshipData]);

  const fetchRelationships = async () => {
    try {
      setLoading(true);
      setError(null);

      const apiUrl = process.env.REACT_APP_API_URL || '';
      const url = `${apiUrl}/api/datasets/${datasetId}/relationships`;

      console.log('RelationshipViewer - Fetching from:', url);
      const response = await fetch(url, {
        timeout: 10000 // 10 second timeout
      });

      if (response.ok) {
        const data = await response.json();
        console.log('RelationshipViewer - Fetched relationship data:', data);
        setRelationships(data);
      } else {
        const errorText = await response.text();
        throw new Error(`Failed to load relationships: ${errorText}`);
      }
    } catch (err) {
      console.error('Error fetching relationships:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
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

  if (error) {
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
  const tableCount = relationships.table_count || 0;
  if (tableCount <= 1) {
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
  const generationOrder = summary.generation_order || [];

  console.log('RelationshipViewer - Rendering with:', {
    totalRelationships,
    relationshipDetails,
    generationOrder,
    tableCount
  });

  return (
    <div className="relationship-viewer">
      <div className="relationship-header">
        <h4>🔗 Table Relationships Detected</h4>
        <div className="relationship-stats">
          <span className="stat-item">
            <strong>{tableCount}</strong> tables
          </span>
          <span className="stat-item">
            <strong>{totalRelationships}</strong> relationships
          </span>
          <span className="stat-item">
            <strong>{summary.tables_with_primary_keys || 0}</strong> with PKs
          </span>
          <span className="stat-item">
            <strong>{summary.tables_with_foreign_keys || 0}</strong> with FKs
          </span>
        </div>
      </div>

      {totalRelationships > 0 ? (
        <>
          {generationOrder.length > 0 && (
            <div className="generation-order">
              <h5>📋 Generation Order</h5>
              <div className="order-flow">
                {generationOrder.map((table, index) => (
                  <React.Fragment key={table}>
                    <div className="table-node">
                      <span className="table-name">{table}</span>
                      <span className="table-order">{index + 1}</span>
                    </div>
                    {index < generationOrder.length - 1 && (
                      <div className="flow-arrow">→</div>
                    )}
                  </React.Fragment>
                ))}
              </div>
              <small className="order-explanation">
                Parent tables are generated first to ensure referential integrity
              </small>
            </div>
          )}

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
          <div style={{
            marginTop: '1rem',
            padding: '1rem',
            background: '#fef3c7',
            borderRadius: '8px',
            textAlign: 'left'
          }}>
            <strong style={{ color: '#92400e' }}>💡 Tips for better relationship detection:</strong>
            <ul style={{ marginTop: '0.5rem', paddingLeft: '1.5rem', color: '#78350f' }}>
              <li>Ensure primary key columns have unique values</li>
              <li>Use consistent naming (e.g., 'customer_id' in both tables)</li>
              <li>Foreign key values should exist in the referenced table</li>
            </ul>
          </div>
        </div>
      )}

      <div className="relationship-notice">
        <h6>ℹ️ How Relationship Detection Works</h6>
        <ul>
          <li><strong>Primary Keys:</strong> Identified by uniqueness (&gt;95%) and naming patterns (*_id, id)</li>
          <li><strong>Foreign Keys:</strong> Detected by value overlap with primary keys (&gt;70%)</li>
          <li><strong>Synthetic Data:</strong> Preserves all detected relationships automatically</li>
          <li><strong>Confidence Scores:</strong> High (&gt;90%) relationships are most reliable</li>
        </ul>
      </div>
    </div>
  );
};

export default RelationshipViewer;
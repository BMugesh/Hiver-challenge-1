import React, { useState } from 'react';

/**
 * Historical Evidence list displaying retrieved cases with similarity & resolution status.
 */
export default function EvidenceList({ evidence = [] }) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  if (!evidence || evidence.length === 0) {
    return (
      <div className="evidence-section">
        <span className="section-micro-label">Historical Evidence</span>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', fontStyle: 'italic' }}>
          No historical evidence retrieved.
        </div>
      </div>
    );
  }

  const toggleExpand = (idx) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <div className="evidence-section">
      <span className="section-micro-label">Historical Evidence ({evidence.length} Cases)</span>
      <div className="evidence-list">
        {evidence.map((item, idx) => {
          const simPct = (item.similarity * 100).toFixed(1);
          const isResolved = item.resolutionStatus === 'CLEARLY_RESOLVED';
          const isExpanded = expandedIndex === idx;

          return (
            <div 
              key={item.caseId || idx} 
              className="evidence-card"
              onClick={() => toggleExpand(idx)}
              style={{ cursor: 'pointer' }}
              title="Click to toggle full historical snippet"
            >
              <div className="evidence-card-header">
                <span className="evidence-case-id">
                  {item.caseId ? `Case #${item.caseId.replace('CASE_', '')}` : `Case #${idx + 1}`}
                </span>
                <div className="evidence-meta-row">
                  <span className="similarity-badge">
                    {item.similarity.toFixed(2)} similarity ({simPct}%)
                  </span>
                  <span className={`status-badge ${isResolved ? 'resolved' : 'partial'}`}>
                    {isResolved ? 'RESOLVED' : 'PARTIAL'}
                  </span>
                </div>
              </div>

              {item.responseText && (
                <div 
                  className="evidence-snippet"
                  style={isExpanded ? { WebkitLineClamp: 'unset', display: 'block' } : {}}
                >
                  {item.responseText}
                </div>
              )}

              {item.resolvedSummary && (
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                  <strong>Outcome:</strong> {item.resolvedSummary}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

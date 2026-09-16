import React from 'react';
import EvidenceList from './EvidenceList';

/**
 * Compact AI Analysis Panel displaying evaluation metrics underneath the AI reply.
 */
export default function AnalysisPanel({ analysis }) {
  if (!analysis) return null;

  const {
    intent = 'GENERAL_DEVICE_INQUIRY',
    intentConfidence = 0.0,
    evidence = [],
    resolutionPattern = '',
    multiIssue = { detected: false, issues: [] },
    promptInjection = { detected: false, attackType: 'NONE' },
    groundingCheck = { status: 'PASS', severity: 'NONE' },
    decision = 'AUTO-HANDLE',
    reason = ''
  } = analysis;

  const isAutoHandle = decision.toUpperCase().includes('AUTO');
  const confPct = Math.round(intentConfidence * 100);
  const isPassGrounding = groundingCheck?.status === 'PASS';
  const isInjectionDetected = promptInjection?.detected === true;
  const isMultiIssueDetected = multiIssue?.detected === true;

  return (
    <div className="analysis-panel">
      {/* Header */}
      <div className="analysis-header">
        <span className="analysis-title">
          <span>⚙</span> AI ANALYSIS
        </span>
        <span className="analysis-badge">
          STAGE 8 PIPELINE
        </span>
      </div>

      {/* 1. Intent & Confidence */}
      <div className="intent-section">
        <div className="intent-label-group">
          <span className="section-micro-label">Classified Intent</span>
          <span className="intent-value">{intent}</span>
        </div>
        <span className="confidence-badge">
          {confPct}% confidence
        </span>
      </div>

      {/* 2. Historical Evidence */}
      <EvidenceList evidence={evidence} />

      {/* 3. Resolution Pattern */}
      {resolutionPattern && (
        <div className="pattern-section">
          <span className="section-micro-label">Resolution Pattern</span>
          <div className="pattern-flow">
            {resolutionPattern}
          </div>
        </div>
      )}

      {/* 4. Multi-Issue Alert if detected */}
      {isMultiIssueDetected && (
        <div className="multi-issue-alert">
          <strong>⚠ Multiple Issues Detected:</strong> Multiple customer issues detected in query. Escalation recommended for human support.
        </div>
      )}

      {/* 5. Evaluation & Defense Checks Grid */}
      <div className="checks-grid">
        <div className="check-item">
          <span className="check-title">Evidence Check</span>
          <span className={`check-status ${isPassGrounding ? 'pass' : 'fail'}`}>
            {isPassGrounding ? '✓ PASS' : '⚠ FAIL'}
          </span>
        </div>

        <div className="check-item">
          <span className="check-title">Prompt Injection</span>
          <span className={`check-status ${isInjectionDetected ? 'fail' : 'pass'}`}>
            {isInjectionDetected ? '⚠ DETECTED' : '✓ NONE'}
          </span>
        </div>

        <div className="check-item">
          <span className="check-title">Multi-Issue Check</span>
          <span className={`check-status ${isMultiIssueDetected ? 'warn' : 'pass'}`}>
            {isMultiIssueDetected ? '⚠ DETECTED' : '✓ NONE'}
          </span>
        </div>
      </div>

      {/* 6. Final Decision Banner */}
      <div className={`decision-banner ${isAutoHandle ? 'auto-handle' : 'escalate'}`}>
        <div className="decision-row">
          <span className="decision-label">Final Decision</span>
          <span className="decision-tag">{decision}</span>
        </div>
        {reason && (
          <div className="decision-reason">
            <strong>Reason:</strong> {reason}
          </div>
        )}
      </div>
    </div>
  );
}

import React from 'react';

/**
 * Page 3 — TRUST & DECISION VIEW
 * Explains why the AI response can be trusted, showing safety gates,
 * prompt injection defense, 7-dimension verification, and policy decisions.
 */
export default function TrustDecisionView({
  analysis,
  conversationId
}) {
  if (!analysis) {
    return (
      <div className="trust-page-container">
        <div className="page-inner-content" style={{ textAlign: 'center', padding: '4rem 1rem' }}>
          <h2 className="page-headline" style={{ justifyContent: 'center' }}>No Active Decision Data</h2>
          <p className="page-subheadline">
            Submit a customer inquiry on the Support tab to observe safety checks and verification decisions.
          </p>
        </div>
      </div>
    );
  }

  const agentDiag = analysis.agentDiagnostics || {};
  const sec = agentDiag.security || {};
  const risk = agentDiag.risk_analysis || {};
  const v = agentDiag.verification || {};
  const dec = agentDiag.decision || {};

  const decision = (analysis.decision || dec.action || 'AUTO-HANDLE').toUpperCase();
  const isAutoHandle = decision.includes('AUTO');
  const isClarify = decision.includes('CLARIFY');
  const isEscalate = !isAutoHandle && !isClarify;

  const isInjectionDetected = analysis.promptInjection?.detected || sec.is_prompt_injection || risk.is_prompt_injection;
  const attackType = analysis.promptInjection?.attackType || sec.security_intent || 'NONE';
  const riskLevel = sec.risk_level || (isInjectionDetected ? 'HIGH' : 'LOW');
  const unsupportedClaims = analysis.groundingCheck?.unsupportedClaims || [];
  const reason = analysis.reason || dec.reason || 'Decision reached based on calibrated policy thresholds.';

  // 7-Point Verification Pass Checks
  const checks = [
    { name: 'Response Relevance', pass: v.relevance === undefined ? true : v.relevance >= 0.70, score: v.relevance },
    { name: 'Evidence Groundedness', pass: v.groundedness === undefined ? analysis.groundingCheck?.status === 'PASS' : v.groundedness, score: null },
    { name: 'Response Actionability', pass: v.actionability === undefined ? true : v.actionability, score: null },
    { name: 'Response Specificity', pass: v.specificity === undefined ? true : v.specificity, score: null },
    { name: 'Resolution Pattern Usage', pass: v.pattern_usage === undefined ? true : v.pattern_usage, score: null },
    { name: 'Anti-Clarification Check', pass: v.unnecessary_clarification === undefined ? true : !v.unnecessary_clarification, score: null },
    { name: 'Safety & Refusal Guard', pass: v.safety_pass === undefined ? !isInjectionDetected : v.safety_pass, score: null }
  ];

  return (
    <div className="trust-page-container">
      <div className="page-inner-content">
        {/* Page Header Banner */}
        <div className="page-header-banner">
          <div className="page-title-group">
            <h1 className="page-headline">
              <span>🛡️</span> TRUST & DECISION
            </h1>
            <p className="page-subheadline">
              Multi-dimensional safety gating, independent claim verification, and deterministic routing.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            {conversationId && (
              <span className="conv-id-badge">{conversationId}</span>
            )}
          </div>
        </div>

        {/* SECTION 1: FINAL DECISION HERO */}
        <div className="decision-hero-banner">
          <div className="decision-hero-main">
            <span className="decision-hero-label">Final Operational Decision</span>
            <div className={`decision-badge-large ${isAutoHandle ? 'auto-handle' : isClarify ? 'clarify' : 'escalate'}`}>
              {isAutoHandle && '✓ AUTO-HANDLE'}
              {isClarify && '○ CLARIFY'}
              {isEscalate && (isInjectionDetected ? '⚠ SAFE REFUSAL + ESCALATE' : '⚠ HUMAN ESCALATION')}
            </div>
            <span className="decision-hero-desc">
              {isAutoHandle
                ? 'Evidence-backed response • Passed 7-dimension verification gate • No high-risk anomalies detected.'
                : isInjectionDetected
                ? 'Security attack intercepted • Neutralized safe response rendered • Incident flagged for escalation.'
                : 'Insufficient evidence for safe autonomous resolution • Case safely queued for Apple Support specialist.'}
            </span>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span className="info-label">Policy Model</span>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Deterministic Gate (v2.0)
            </div>
          </div>
        </div>

        {/* SECTION 2: SAFETY & SECURITY (DECOUPLED BUSINESS VS SECURITY) */}
        <section className="intelligence-section-card" aria-labelledby="safety-checks-title">
          <div className="section-title-row">
            <h2 id="safety-checks-title" className="section-heading">
              <span>02</span> Safety & Security Checks
            </h2>
            <span className={`status-chip ${isInjectionDetected ? 'danger' : 'verified'}`}>
              {isInjectionDetected ? 'SECURITY ALERT' : 'ALL CHECKS PASSED'}
            </span>
          </div>

          <div className="security-business-grid">
            {/* Business Intent Classification */}
            <div className="intent-classification-card">
              <span className="card-micro-header">Business Intent Domain</span>
              <div style={{ fontSize: '1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-blue)' }}>
                {analysis.intent || 'GENERAL_DEVICE_INQUIRY'}
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Standard 11-intent Apple customer support taxonomy classification.
              </p>
            </div>

            {/* Independent Security Intent Analysis */}
            <div className={`security-risk-card ${isInjectionDetected ? 'danger' : ''}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-micro-header">Security Risk Analysis</span>
                <span className={`status-chip ${isInjectionDetected ? 'danger' : 'verified'}`}>
                  Risk: {riskLevel}
                </span>
              </div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: isInjectionDetected ? 'var(--status-red)' : 'var(--status-green)' }}>
                {isInjectionDetected ? `ATTACK: ${attackType}` : 'NONE DETECTED ✓'}
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                {isInjectionDetected
                  ? 'Adversarial instruction detected and separated from business domain.'
                  : 'No adversarial prompt-injection, jailbreak, or exfiltration patterns detected.'}
              </p>
            </div>
          </div>

          {/* Unsupported Claims Summary */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.65rem 0.85rem', background: 'var(--bg-surface-secondary)', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Unsupported Claims Count
            </span>
            <span className={`status-chip ${unsupportedClaims.length === 0 ? 'verified' : 'danger'}`}>
              {unsupportedClaims.length === 0 ? '0 Unsupported Claims ✓' : `${unsupportedClaims.length} Claims Flagged ⚠`}
            </span>
          </div>
        </section>

        {/* SECTION 3: 7-DIMENSION RESPONSE VERIFICATION */}
        <section className="intelligence-section-card" aria-labelledby="verification-matrix-title">
          <div className="section-title-row">
            <h2 id="verification-matrix-title" className="section-heading">
              <span>03</span> Response Verification Matrix
            </h2>
            <span className="section-badge intent-chip">Independent Gate</span>
          </div>

          <div className="verification-matrix-grid">
            {checks.map((c, idx) => (
              <div key={idx} className="verification-check-card">
                <span className="verification-check-name">{c.name}</span>
                <span className={`verification-status-tag ${c.pass ? 'pass' : 'fail'}`}>
                  {c.pass ? 'PASS ✓' : 'FAIL ⚠'}
                </span>
              </div>
            ))}
          </div>

          {v.failure_reasons && v.failure_reasons.length > 0 && (
            <div style={{ padding: '0.6rem 0.85rem', background: 'var(--status-red-bg)', border: '1px solid var(--status-red-border)', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem', color: 'var(--status-red)' }}>
              <strong>Verification Interceptions:</strong>
              <ul style={{ paddingLeft: '1.2rem', marginTop: '0.25rem' }}>
                {v.failure_reasons.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {/* SECTION 4: HUMAN-READABLE DECISION REASON */}
        <section className="intelligence-section-card" aria-labelledby="decision-reason-title">
          <div className="section-title-row">
            <h2 id="decision-reason-title" className="section-heading">
              <span>04</span> Decision Reason & Audit Log
            </h2>
            <span className="section-badge intent-chip">{analysis.reasonCode || 'DECISION_POLICY_EVALUATED'}</span>
          </div>

          <div className="decision-reason-callout">
            {reason}
          </div>
        </section>
      </div>
    </div>
  );
}

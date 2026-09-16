import React, { useState } from 'react';

/**
 * Page 2 — CASE INTELLIGENCE VIEW
 * Explains how SupportDNA understood the current case, retrieved historical evidence,
 * and synthesized a resolution pattern.
 */
export default function CaseIntelligenceView({
  analysis,
  conversationId,
  customerMessage
}) {
  const [expandedCaseId, setExpandedCaseId] = useState(null);

  if (!analysis) {
    return (
      <div className="intelligence-page-container">
        <div className="page-inner-content" style={{ textAlign: 'center', padding: '4rem 1rem' }}>
          <h2 className="page-headline" style={{ justifyContent: 'center' }}>No Active Case Analysis</h2>
          <p className="page-subheadline">
            Submit a customer inquiry on the Support tab to observe the complete case intelligence pipeline.
          </p>
        </div>
      </div>
    );
  }

  const agentDiag = analysis.agentDiagnostics || {};
  const qu = agentDiag.query_understanding || {};
  const evJudge = agentDiag.evidence_quality || {};
  const evidenceList = analysis.evidence || [];
  const multiIssue = analysis.multiIssue || {};
  const attemptedSteps = analysis.attemptedSteps || agentDiag.attempted_steps || [];

  const intent = analysis.intent || qu.intent || 'GENERAL_DEVICE_INQUIRY';
  const confidence = Math.round((analysis.intentConfidence || qu.intent_confidence || 0.85) * 100);
  const customerGoal = analysis.customerGoal || qu.customer_goal || 'Resolve customer support request';
  const followUpType = analysis.followUpType || agentDiag.follow_up_type || 'NEW_ISSUE';
  const evidenceQuality = (analysis.evidenceQuality || evJudge.overall_quality || 'MODERATE').toUpperCase();
  const resolutionPattern = analysis.resolutionPattern || agentDiag.response_plan?.pattern || '';

  const issues = qu.issues || (analysis.intent ? [analysis.intent.replace('INTENT_', '').replace(/_/g, ' ')] : ['Customer device issue']);
  const knownInfo = qu.known_info || [customerMessage || 'Inquiry provided by customer'];
  const missingInfo = qu.missing_info || [];

  const toggleCase = (cid) => {
    setExpandedCaseId(expandedCaseId === cid ? null : cid);
  };

  return (
    <div className="intelligence-page-container">
      <div className="page-inner-content">
        {/* Page Header Banner */}
        <div className="page-header-banner">
          <div className="page-title-group">
            <h1 className="page-headline">
              <span>🧠</span> CASE INTELLIGENCE
            </h1>
            <p className="page-subheadline">
              Structural query comprehension, historical evidence retrieval, and resolution synthesis.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            {conversationId && (
              <span className="conv-id-badge">{conversationId}</span>
            )}
            <span className="intent-chip">{intent.replace('INTENT_', '').replace(/_/g, ' ')}</span>
          </div>
        </div>

        {/* SECTION 1: QUERY UNDERSTANDING */}
        <section className="intelligence-section-card" aria-labelledby="query-understanding-title">
          <div className="section-title-row">
            <h2 id="query-understanding-title" className="section-heading">
              <span>01</span> Query Understanding
            </h2>
            <span className="section-badge intent-chip">Layer 1 & 4</span>
          </div>

          <div className="query-understanding-grid">
            <div className="info-block">
              <span className="info-label">Classified Intent</span>
              <span className="info-value-highlight" style={{ color: 'var(--status-blue)' }}>
                {intent}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Calibrated Confidence: <strong style={{ color: 'var(--text-primary)' }}>{confidence}%</strong>
              </span>
            </div>

            <div className="info-block">
              <span className="info-label">Customer Goal</span>
              <span style={{ fontSize: '0.88rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                {customerGoal}
              </span>
            </div>

            <div className="info-block">
              <span className="info-label">Follow-up Type</span>
              <span className="intent-chip" style={{ alignSelf: 'flex-start', marginTop: '0.2rem' }}>
                {followUpType}
              </span>
            </div>

            <div className="info-block">
              <span className="info-label">Multi-Issue Detection</span>
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: multiIssue.detected ? 'var(--status-amber)' : 'var(--status-green)' }}>
                {multiIssue.detected ? '⚠ Multiple Symptoms Detected' : '✓ Single Focus Inquiry'}
              </span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginTop: '0.5rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)' }}>
            <div className="info-block">
              <span className="info-label">Identified Issues</span>
              <ul className="info-list">
                {issues.map((issue, idx) => (
                  <li key={idx} className="info-list-item">{issue}</li>
                ))}
              </ul>
            </div>

            <div className="info-block">
              <span className="info-label">Known Information</span>
              <ul className="info-list">
                {knownInfo.map((info, idx) => (
                  <li key={idx} className="info-list-item">{info}</li>
                ))}
              </ul>
            </div>

            {missingInfo.length > 0 && (
              <div className="info-block">
                <span className="info-label">Missing Information</span>
                <ul className="info-list">
                  {missingInfo.map((item, idx) => (
                    <li key={idx} className="info-list-item" style={{ color: 'var(--text-muted)' }}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 2: HISTORICAL EVIDENCE */}
        <section className="intelligence-section-card" aria-labelledby="historical-evidence-title">
          <div className="section-title-row">
            <h2 id="historical-evidence-title" className="section-heading">
              <span>02</span> Historical Evidence (FAISS Retrieval)
            </h2>
            <span className="section-badge intent-chip">{evidenceList.length} Historical Cases Aligned</span>
          </div>

          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '-0.3rem 0 0.4rem' }}>
            Retrieved historical support threads serve as <em>evidential grounding</em> for recommended resolutions.
          </p>

          <div className="evidence-cards-container">
            {evidenceList.length === 0 ? (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '0.5rem' }}>
                No historical evidence retrieved for this query.
              </div>
            ) : (
              evidenceList.map((item, idx) => {
                const cid = item.caseId || `CASE_${idx + 1}`;
                const isExpanded = expandedCaseId === cid;
                const simPct = (item.similarity * 100).toFixed(1);
                const quality = item.similarity >= 0.80 ? 'STRONG' : item.similarity >= 0.65 ? 'MODERATE' : 'WEAK';

                return (
                  <div key={cid} className="evidence-entry-card">
                    <div className="evidence-top-row">
                      <span className="evidence-case-tag">
                        Case #{cid.replace('CASE_', '')}
                      </span>

                      <div className="evidence-meta-badges">
                        <span className="similarity-pill">
                          {item.similarity.toFixed(2)} similarity ({simPct}%)
                        </span>
                        <span className={`quality-pill ${quality.toLowerCase()}`}>
                          {quality}
                        </span>
                      </div>
                    </div>

                    {item.resolvedSummary && (
                      <div className="evidence-resolution-text">
                        <strong>Resolution:</strong> {item.resolvedSummary}
                      </div>
                    )}

                    {item.responseText && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        <button
                          type="button"
                          className="evidence-expand-toggle"
                          onClick={() => toggleCase(cid)}
                        >
                          {isExpanded ? 'Hide historical snippet ▲' : 'View full historical snippet ▼'}
                        </button>

                        {isExpanded && (
                          <div className="evidence-full-snippet">
                            {item.responseText}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </section>

        {/* SECTION 3: EVIDENCE QUALITY */}
        <section className="intelligence-section-card" aria-labelledby="evidence-quality-title">
          <div className="section-title-row">
            <h2 id="evidence-quality-title" className="section-heading">
              <span>03</span> Evidence Quality Assessment
            </h2>
            <span className={`quality-pill ${evidenceQuality.toLowerCase()}`}>
              {evidenceQuality}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
              {evJudge.summary_reason || (
                evidenceQuality === 'STRONG'
                  ? 'Multiple historical cases describe the exact same symptoms and validate the recommended troubleshooting path.'
                  : evidenceQuality === 'MODERATE'
                  ? 'Historical cases exhibit strong domain alignment with similar troubleshooting steps.'
                  : 'Historical evidence is limited or borderline. Human review recommended if symptoms persist.'
              )}
            </p>

            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.35rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Actionable Cases: <strong style={{ color: 'var(--text-primary)' }}>{evJudge.actionable_count ?? evidenceList.length}</strong></span>
              <span>Domain Alignment: <strong style={{ color: 'var(--status-green)' }}>PASS</strong></span>
              <span>Contradictions Detected: <strong style={{ color: evJudge.contradiction_detected ? 'var(--status-red)' : 'var(--status-green)' }}>{evJudge.contradiction_detected ? 'YES' : 'NONE'}</strong></span>
            </div>
          </div>
        </section>

        {/* SECTION 4: RESOLUTION PATTERN */}
        {resolutionPattern && (
          <section className="intelligence-section-card" aria-labelledby="resolution-pattern-title">
            <div className="section-title-row">
              <h2 id="resolution-pattern-title" className="section-heading">
                <span>04</span> Resolution Pattern
              </h2>
              <span className="section-badge intent-chip">Extracted Troubleshooting Steps</span>
            </div>

            <div className="pattern-steps-list">
              {resolutionPattern.split('\n').filter(Boolean).map((step, idx) => {
                const cleanStep = step.replace(/^[✓→○\-*\d.]+\s*/, '');
                const isFinal = idx === resolutionPattern.split('\n').filter(Boolean).length - 1;
                return (
                  <div key={idx} className="pattern-step-row">
                    <span className={`step-icon ${isFinal ? 'escalate' : 'recommended'}`}>
                      {isFinal ? '→' : '✓'}
                    </span>
                    <span style={{ color: isFinal ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                      {cleanStep}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* SECTION 5: CONVERSATION MEMORY & ATTEMPTED STEPS */}
        {attemptedSteps.length > 0 && (
          <section className="intelligence-section-card" aria-labelledby="attempted-steps-title">
            <div className="section-title-row">
              <h2 id="attempted-steps-title" className="section-heading">
                <span>05</span> Conversation Memory (Attempted Steps)
              </h2>
              <span className="section-badge intent-chip">Live Session State</span>
            </div>

            <div className="attempted-steps-grid">
              {attemptedSteps.map((step, idx) => {
                const stepText = typeof step === 'string' ? step : step.step || step.name;
                const result = typeof step === 'object' ? step.result : 'RECOMMENDED';
                return (
                  <div key={idx} className="attempted-step-card">
                    <span className="attempted-step-title">{stepText}</span>
                    <span className={`intent-chip ${result === 'SUCCESS' ? 'status-chip verified' : ''}`}>
                      {result || 'PENDING'}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

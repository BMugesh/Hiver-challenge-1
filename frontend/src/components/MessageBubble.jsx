import React, { useState } from 'react';

/**
 * Clean Message Bubble with Hero Answer and Compact Response Status Row
 */
export default function MessageBubble({ message, onInspectIntelligence }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="message-row user">
        <div className="message-header-row">
          <span>Customer</span>
          <span>&bull;</span>
          <span>{message.timestamp || 'Just now'}</span>
        </div>
        <div className="user-bubble">
          {message.content}
        </div>
      </div>
    );
  }

  const analysis = message.analysis || {};
  const decision = (analysis.decision || 'AUTO-HANDLE').toUpperCase();
  const isAutoHandle = decision.includes('AUTO');
  const groundingPass = analysis.groundingCheck?.status === 'PASS';
  const injectionDetected = analysis.promptInjection?.detected === true;
  const isMultiIssue = analysis.multiIssue?.detected === true;

  const handleCopy = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <div className="message-row ai">
      <div className="message-header-row">
        <span> SupportDNA</span>
        <span>&bull;</span>
        <span>{message.timestamp || 'Just now'}</span>
      </div>

      <div className="ai-response-wrapper">
        {/* Hero AI Answer */}
        <div className="ai-bubble">
          <div className="ai-reply-text">
            {message.content}
          </div>

          <div className="ai-bubble-actions">
            <button
              type="button"
              className="action-icon-btn"
              onClick={handleCopy}
              title="Copy response to clipboard"
            >
              {copied ? '✓ Copied' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Compact Metadata Chips */}
        <div className="ai-status-row">
          {groundingPass && (
            <span className="status-chip verified" title="Response grounded in verified historical cases">
              ✓ Evidence-backed
            </span>
          )}

          <span className={`status-chip ${isAutoHandle ? 'verified' : 'danger'}`} title="Decision Action Policy">
            {isAutoHandle ? '✓ AUTO-HANDLE' : '⚠ ESCALATE'}
          </span>

          {injectionDetected && (
            <span className="status-chip danger" title="Prompt injection attack neutralized">
              ⚠ Prompt Injection
            </span>
          )}

          {isMultiIssue && (
            <span className="status-chip warn" title="Multiple customer symptoms detected">
              ⚠ Multi-Issue
            </span>
          )}

          {analysis.intent && (
            <span className="status-chip info" title="Classified business intent">
              {analysis.intent.replace('INTENT_', '').replace(/_/g, ' ')}
            </span>
          )}

          {onInspectIntelligence && (
            <button
              type="button"
              className="explore-intelligence-btn"
              onClick={() => onInspectIntelligence(analysis)}
              title="Inspect complete query understanding and historical evidence"
            >
              Inspect Case Intelligence &rarr;
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

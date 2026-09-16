import React from 'react';

/**
 * Minimal Top Header and Navigation Bar
 */
export default function TopHeader({
  activeTab,
  onSelectTab,
  conversationId,
  onNewChat
}) {
  return (
    <header className="top-header">
      {/* Left: Branding */}
      <div className="header-brand">
        <div className="brand-logo-badge">
          <span></span>
        </div>
        <div className="brand-info">
          <div className="brand-title">
            SupportDNA
            <span className="brand-version-pill">v2.0</span>
          </div>
          <span className="brand-subtitle">AI Support Intelligence</span>
        </div>
      </div>

      {/* Center: Compact Navigation Tabs */}
      <nav className="header-nav" aria-label="Main Navigation">
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'support' ? 'active' : ''}`}
          onClick={() => onSelectTab('support')}
        >
          <span>Support</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'intelligence' ? 'active' : ''}`}
          onClick={() => onSelectTab('intelligence')}
        >
          <span>Case Intelligence</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'trust' ? 'active' : ''}`}
          onClick={() => onSelectTab('trust')}
        >
          <span>Trust & Decision</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
          onClick={() => onSelectTab('evaluation')}
        >
          <span>Evaluation</span>
          <span className="nav-tab-badge">N=1,189</span>
        </button>
      </nav>

      {/* Right: Status Pill & Actions */}
      <div className="header-right">
        <div className="system-status-pill" title="Live 9-Stage Knowledge Pipeline Connected">
          <span className="status-pulse-dot" />
          <span>Pipeline Active</span>
        </div>

        {conversationId && (
          <button
            type="button"
            className="header-btn"
            onClick={onNewChat}
            title="Start a new support conversation"
          >
            <span>+</span> New Chat
          </button>
        )}
      </div>
    </header>
  );
}

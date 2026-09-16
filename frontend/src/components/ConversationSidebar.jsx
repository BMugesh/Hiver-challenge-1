import React, { useState } from 'react';

/**
 * 25-30% Conversation List Sidebar for Session Memory
 */
export default function ConversationSidebar({
  conversations = [],
  activeConversationId,
  onSelectConversation,
  onNewChat
}) {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = conversations.filter((c) => {
    if (!searchTerm) return true;
    const term = searchTerm.toLowerCase();
    const title = (c.title || c.current_customer_goal || c.conversation_id || '').toLowerCase();
    const intent = (c.current_intent || '').toLowerCase();
    return title.includes(term) || intent.includes(term);
  });

  const formatTimestamp = (isoStr) => {
    if (!isoStr) return 'Just now';
    try {
      const date = new Date(isoStr);
      const now = new Date();
      const diffMin = Math.floor((now - date) / (1000 * 60));
      if (diffMin < 2) return 'Just now';
      if (diffMin < 60) return `${diffMin}m ago`;
      const diffHours = Math.floor(diffMin / 60);
      if (diffHours < 24) return `${diffHours}h ago`;
      return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch {
      return 'Recent';
    }
  };

  return (
    <aside className="conversation-sidebar" aria-label="Conversation History">
      <div className="sidebar-header">
        <div className="sidebar-top-row">
          <span className="sidebar-title">Conversations</span>
          <button
            type="button"
            className="new-chat-btn"
            onClick={onNewChat}
            title="Start fresh conversation"
          >
            <span>+</span> New Chat
          </button>
        </div>

        <div className="sidebar-search-box">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-dim)' }}>
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input
            type="text"
            className="sidebar-search-input"
            placeholder="Search conversations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="sidebar-conversations-list">
        {filtered.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            {searchTerm ? 'No matching conversations' : 'No previous conversations'}
          </div>
        ) : (
          filtered.map((conv) => {
            const isActive = conv.conversation_id === activeConversationId;
            const title = conv.title || conv.current_customer_goal || (conv.conversation_id ? `Case #${conv.conversation_id.slice(-6)}` : 'Support Inquiry');
            const preview = conv.preview || conv.last_message || 'Active conversation session';
            const intent = conv.current_intent ? conv.current_intent.replace('INTENT_', '').replace(/_/g, ' ') : null;

            return (
              <button
                key={conv.conversation_id}
                type="button"
                className={`conversation-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.conversation_id)}
              >
                <div className="conv-item-top">
                  <span className="conv-item-title" title={title}>{title}</span>
                  <span className="conv-item-time">{formatTimestamp(conv.updated_at || conv.created_at)}</span>
                </div>
                <span className="conv-item-preview">{preview}</span>
                {intent && (
                  <div className="conv-item-footer">
                    <span className="intent-chip">{intent}</span>
                  </div>
                )}
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}

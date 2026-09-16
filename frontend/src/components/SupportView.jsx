import React, { useEffect, useRef } from 'react';
import ConversationSidebar from './ConversationSidebar';
import MessageBubble from './MessageBubble';
import InputBox from './InputBox';

const EMPTY_SUGGESTIONS = [
  { tag: 'Battery', text: 'My iPhone battery is draining very fast after updating to iOS 11.' },
  { tag: 'WiFi', text: 'My iPhone keeps dropping WiFi connection and won\'t reconnect.' },
  { tag: 'Keyboard', text: 'When I type the letter I it replaces it with an exclamation question mark box.' },
  { tag: 'Billing', text: 'I was charged twice for an in-app subscription in the App Store.' }
];

/**
 * Page 1 — Primary Support Workspace (2-Pane View)
 */
export default function SupportView({
  conversations,
  activeConversationId,
  messages,
  isLoading,
  error,
  latestAnalysis,
  onSendMessage,
  onSelectConversation,
  onNewChat,
  onRetry,
  onInspectIntelligence
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, error]);

  const isEmpty = messages.length === 0;
  const currentIntent = latestAnalysis?.intent
    ? latestAnalysis.intent.replace('INTENT_', '').replace(/_/g, ' ')
    : null;

  return (
    <div className="support-workspace">
      {/* Left 25-30%: Conversation History Sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
        onNewChat={onNewChat}
      />

      {/* Right 70-75%: Main Conversation Stream */}
      <main className="conversation-main" aria-label="Support Conversation Workspace">
        {/* Workspace Sub-Header */}
        <div className="conversation-header">
          <div className="conv-header-title-group">
            <span className="conv-header-title">Support Conversation</span>
            {activeConversationId && (
              <span className="conv-id-badge" title="Active Session ID">
                {activeConversationId}
              </span>
            )}
          </div>

          <div className="conv-header-meta">
            {currentIntent && (
              <span className="intent-chip" title="Current Detected Issue Intent">
                {currentIntent}
              </span>
            )}
          </div>
        </div>

        {/* Message Stream */}
        <div className="chat-stream-container">
          {isEmpty ? (
            <div className="chat-empty-state">
              <div className="empty-state-badge"></div>
              <h2 className="empty-state-title">SupportDNA Workspace</h2>
              <p className="empty-state-desc">
                An intelligent customer support agent grounded in historical resolution evidence.
                Ask any customer support question or choose a sample case below to observe reasoning in real time.
              </p>

              <div className="empty-suggestions-group">
                <span className="suggestions-heading">Sample Support Inquiries</span>
                <div className="suggestions-cards-grid">
                  {EMPTY_SUGGESTIONS.map((item, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="suggestion-card-btn"
                      onClick={() => onSendMessage(item.text)}
                    >
                      <span className="suggestion-tag">{item.tag}</span>
                      <span className="suggestion-text">{item.text}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => (
                <MessageBubble
                  key={msg.id || idx}
                  message={msg}
                  onInspectIntelligence={onInspectIntelligence}
                />
              ))}

              {error && (
                <div className="error-banner" role="alert">
                  <span>{error}</span>
                  {onRetry && (
                    <button type="button" className="error-retry-btn" onClick={onRetry}>
                      Retry
                    </button>
                  )}
                </div>
              )}

              <div ref={bottomRef} style={{ height: 1 }} />
            </>
          )}
        </div>

        {/* Bottom Input Area */}
        <InputBox onSendMessage={onSendMessage} isLoading={isLoading} />
      </main>
    </div>
  );
}

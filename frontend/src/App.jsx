import React, { useState, useEffect } from 'react';
import TopHeader from './components/TopHeader';
import SupportView from './components/SupportView';
import CaseIntelligenceView from './components/CaseIntelligenceView';
import TrustDecisionView from './components/TrustDecisionView';
import EvaluationView from './components/EvaluationView';

const API_BASE_URL = 'http://127.0.0.1:8000';

const generateConversationId = () => `conv_${Math.random().toString(36).substring(2, 9)}`;

export default function App() {
  const [activeTab, setActiveTab] = useState('support');
  const [conversationId, setConversationId] = useState(generateConversationId);
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [latestAnalysis, setLatestAnalysis] = useState(null);
  const [lastCustomerQuery, setLastCustomerQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // 1. Fetch recent conversation list on initial mount
  useEffect(() => {
    let isMounted = true;
    fetch(`${API_BASE_URL}/api/conversations`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (isMounted && Array.isArray(data) && data.length > 0) {
          setConversations(data);
        }
      })
      .catch(() => {
        // Backend not yet ready or offline; will populate from local state
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Start a fresh conversation
  const handleNewChat = () => {
    const newId = generateConversationId();
    setConversationId(newId);
    setMessages([]);
    setLatestAnalysis(null);
    setLastCustomerQuery('');
    setError(null);
    setActiveTab('support');
  };

  // 3. Select an existing conversation from sidebar
  const handleSelectConversation = async (convId) => {
    if (convId === conversationId) return;
    setConversationId(convId);
    setError(null);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/conversations/${convId}`);
      if (res.ok) {
        const data = await res.json();
        const convMsgs = (data.messages || []).map((m) => ({
          id: `msg-${m.id || Date.now()}`,
          role: m.role.toLowerCase() === 'assistant' ? 'ai' : 'user',
          content: m.content,
          timestamp: m.created_at ? new Date(m.created_at).toLocaleTimeString() : 'Recent'
        }));
        setMessages(convMsgs);

        // Find last user query
        const lastUser = convMsgs.filter((m) => m.role === 'user').pop();
        if (lastUser) setLastCustomerQuery(lastUser.content);
      }
    } catch (err) {
      console.error('Failed to load conversation history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // 4. Send customer inquiry to backend pipeline
  const handleSendMessage = async (queryText) => {
    if (!queryText || !queryText.trim() || isLoading) return;

    const trimmedQuery = queryText.trim();
    setLastCustomerQuery(trimmedQuery);
    setError(null);

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmedQuery,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/support/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: trimmedQuery,
          conversationId: conversationId
        })
      });

      if (!response.ok) {
        throw new Error(`Pipeline API Error: ${response.status}`);
      }

      const data = await response.json();

      if (data.conversationId && data.conversationId !== conversationId) {
        setConversationId(data.conversationId);
      }

      setLatestAnalysis(data);

      const aiMsg = {
        id: `ai-${Date.now()}`,
        role: 'ai',
        content: data.reply || 'Unable to generate response.',
        analysis: data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, aiMsg]);

      // Update conversations sidebar list locally
      setConversations((prev) => {
        const existingIdx = prev.findIndex((c) => c.conversation_id === (data.conversationId || conversationId));
        const updatedEntry = {
          conversation_id: data.conversationId || conversationId,
          title: data.analysis?.customerGoal || trimmedQuery.slice(0, 32),
          current_intent: data.intent,
          last_message: trimmedQuery,
          updated_at: new Date().toISOString()
        };

        if (existingIdx >= 0) {
          const updated = [...prev];
          updated[existingIdx] = { ...updated[existingIdx], ...updatedEntry };
          return updated;
        } else {
          return [updatedEntry, ...prev];
        }
      });
    } catch (err) {
      console.error('SupportDNA pipeline error:', err);
      setError('SupportDNA pipeline connection unavailable. Please ensure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastCustomerQuery) {
      handleSendMessage(lastCustomerQuery);
    }
  };

  const handleInspectIntelligence = (analysisObj) => {
    if (analysisObj) setLatestAnalysis(analysisObj);
    setActiveTab('intelligence');
  };

  return (
    <div className="app-shell">
      {/* Top Header & Navigation */}
      <TopHeader
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        conversationId={conversationId}
        onNewChat={handleNewChat}
      />

      {/* Main Multi-View Area */}
      <div className="main-view-container">
        {activeTab === 'support' && (
          <SupportView
            conversations={conversations}
            activeConversationId={conversationId}
            messages={messages}
            isLoading={isLoading}
            error={error}
            latestAnalysis={latestAnalysis}
            onSendMessage={handleSendMessage}
            onSelectConversation={handleSelectConversation}
            onNewChat={handleNewChat}
            onRetry={handleRetry}
            onInspectIntelligence={handleInspectIntelligence}
          />
        )}

        {activeTab === 'intelligence' && (
          <CaseIntelligenceView
            analysis={latestAnalysis}
            conversationId={conversationId}
            customerMessage={lastCustomerQuery}
          />
        )}

        {activeTab === 'trust' && (
          <TrustDecisionView
            analysis={latestAnalysis}
            conversationId={conversationId}
          />
        )}

        {activeTab === 'evaluation' && (
          <EvaluationView />
        )}
      </div>
    </div>
  );
}

import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useParams, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import SettingsModal from './components/SettingsModal';
import { ThemeProvider } from './contexts/ThemeContext';
import { api } from './api';
import './App.css';

function ConversationView() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [loadingConversations, setLoadingConversations] = useState({});
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const setConversationLoading = useCallback((convId, loading) => {
    if (!convId) return;
    setLoadingConversations((prev) => {
      if (loading) {
        if (prev[convId]) return prev;
        return { ...prev, [convId]: true };
      }
      if (!prev[convId]) return prev;
      const next = { ...prev };
      delete next[convId];
      return next;
    });
  }, []);
  const isCurrentConversationLoading = conversationId ? !!loadingConversations[conversationId] : false;

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load conversation details when URL changes
  useEffect(() => {
    if (conversationId) {
      loadConversation(conversationId);
    } else {
      setCurrentConversation(null);
    }
  }, [conversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      navigate(`/c/${newConv.id}`); // Navigate to new conversation URL
      setSidebarOpen(false); // Close sidebar on mobile after creating new conversation
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    navigate(`/c/${id}`); // Navigate to conversation URL
    setSidebarOpen(false); // Close sidebar on mobile after selecting
  };

  const handleDeleteConversation = async (convId) => {
    if (!confirm('Are you sure you want to delete this conversation?')) {
      return;
    }

    try {
      await api.deleteConversation(convId);

      // Remove from conversations list
      setConversations(conversations.filter(c => c.id !== convId));
      setConversationLoading(convId, false);

      // If current conversation was deleted, navigate to home
      if (conversationId === convId) {
        navigate('/');
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      alert('Failed to delete conversation');
    }
  };

  // Check for pending jobs on mount and resume polling
  useEffect(() => {
    if (!conversationId) return;

    const pendingJobs = JSON.parse(localStorage.getItem('pendingJobs') || '{}');
    const jobsForConv = Object.entries(pendingJobs).filter(([_, convId]) => convId === conversationId);

    if (jobsForConv.length > 0) {
      // Set loading state for this conversation
      setConversationLoading(conversationId, true);

      jobsForConv.forEach(([jobId, convId]) => {
        pollJobStatus(jobId, convId);
      });
    }
  }, [conversationId]);

  const pollJobStatus = async (jobId, conversationId) => {
    const pollInterval = setInterval(async () => {
      try {
        const job = await api.getJobStatus(jobId);

        if (job.status === 'completed') {
          clearInterval(pollInterval);
          // Remove from pending jobs
          const pendingJobs = JSON.parse(localStorage.getItem('pendingJobs') || '{}');
          delete pendingJobs[jobId];
          localStorage.setItem('pendingJobs', JSON.stringify(pendingJobs));

          // Reload conversation to show result
          if (conversationId) {
            await loadConversation(conversationId);
          }
          setConversationLoading(conversationId, false);
          loadConversations();
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          // Remove from pending jobs
          const pendingJobs = JSON.parse(localStorage.getItem('pendingJobs') || '{}');
          delete pendingJobs[jobId];
          localStorage.setItem('pendingJobs', JSON.stringify(pendingJobs));

          // Only show error if not cancelled by user
          if (job.error !== 'Cancelled by user') {
            console.error('Job failed:', job.error);
          }

          // Reload conversation to update UI
          if (conversationId) {
            await loadConversation(conversationId);
          }

          setConversationLoading(conversationId, false);
        } else {
          // Still processing - update UI
          if (conversationId) {
            setCurrentConversation((prev) => {
              if (!prev || prev.id !== conversationId) return prev;
              const messages = [...prev.messages];
              // Find the message with this jobId
              const msgIndex = messages.findIndex(m => m.role === 'assistant' && m.jobId === jobId);
              if (msgIndex !== -1) {
                messages[msgIndex].jobStatus = job.status;
              } else {
                // Job exists but message not found (page was reloaded)
                // Add a placeholder message
                if (!messages.some(m => m.jobId === jobId)) {
                  messages.push({
                    role: 'assistant',
                    stage1: null,
                    stage2: null,
                    stage3: null,
                    metadata: null,
                    jobStatus: job.status,
                    jobId: jobId,
                  });
                }
              }
              return { ...prev, messages };
            });
          }
        }
      } catch (error) {
        console.error('Failed to poll job status:', error);
      }
    }, 3000); // Poll every 3 seconds
  };

  const handleSendMessage = async (content) => {
    if (!conversationId) return;

    setConversationLoading(conversationId, true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Send message for async processing first
      const { job_id } = await api.sendMessageAsync(conversationId, content);

      // Create a partial assistant message showing it's processing
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        jobStatus: 'pending',
        jobId: job_id,  // Store job_id in message
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Store job_id in localStorage for recovery after browser close
      const pendingJobs = JSON.parse(localStorage.getItem('pendingJobs') || '{}');
      pendingJobs[job_id] = conversationId;
      localStorage.setItem('pendingJobs', JSON.stringify(pendingJobs));

      // Start polling for job status
      pollJobStatus(job_id, conversationId);

    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setConversationLoading(conversationId, false);
    }
  };

  const handleSendMessageOld = async (content) => {
    if (!conversationId) return;

    setConversationLoading(conversationId, true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage3: false,
        },
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(conversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = event.data;
              lastMsg.loading.stage1 = false;
              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2 = event.data;
              lastMsg.metadata = event.metadata;
              lastMsg.loading.stage2 = false;
              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage3 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3 = event.data;
              lastMsg.loading.stage3 = false;
              return { ...prev, messages };
            });
            break;

          case 'title_complete':
            // Reload conversations to get updated title
            loadConversations();
            break;

          case 'complete':
            // Stream complete, reload conversations list
            loadConversations();
            setConversationLoading(conversationId, false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setConversationLoading(conversationId, false);
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setConversationLoading(conversationId, false);
    }
  };

  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <ThemeProvider>
      <div className="app">
        {/* Hamburger menu button - mobile only */}
        <button
          className={`hamburger-menu ${sidebarOpen ? 'open' : ''}`}
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label="Toggle menu"
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        <Sidebar
          conversations={conversations}
          currentConversationId={conversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
        <ChatInterface
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          isLoading={isCurrentConversationLoading}
        />
        <SettingsModal
          isOpen={settingsOpen}
          onClose={() => setSettingsOpen(false)}
        />
      </div>
    </ThemeProvider>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Routes>
        <Route path="/" element={<ConversationView />} />
        <Route path="/c/:conversationId" element={<ConversationView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ThemeProvider>
  );
}

export default App;

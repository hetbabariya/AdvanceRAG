import * as React from 'react';
import { chatApi, Citation } from '../api';
import MarkdownRenderer from './MarkdownRenderer';
import './ChatInterface.css';

const { useState, useRef, useEffect } = React;

interface ChatInterfaceProps {
    fileName: string;
}

interface ChatMessage {
    id: string;
    sender: 'user' | 'ai';
    text: string;
    citations?: Citation[];
    cached?: boolean;
}

type AgentActivityItem = {
    id: string;
    ts: number;
    event: string;
    step?: number;
    max_steps?: number;
    query?: string;
    previous_query?: string;
    docs?: number;
    top_k?: number;
    reason?: string;
    tool_calls?: string[];
    latency_s?: number;
};

const ChatInterface: React.FC<ChatInterfaceProps> = ({ fileName }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [agentMode, setAgentMode] = useState(false);
    const [agentActivity, setAgentActivity] = useState<AgentActivityItem[]>([]);
    const [activityOpen, setActivityOpen] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const formatAgentEvent = (e: AgentActivityItem) => {
        const labelMap: Record<string, string> = {
            step_start: 'Step started',
            agent_decision: 'Agent decided',
            rewrite: 'Rewrite query',
            decompose: 'Decompose question',
            retrieve: 'Retrieve docs',
            rerank: 'Rerank',
            adaptive_top_k: 'Adaptive top-k',
            tool_validation_error: 'Tool validation',
            answer_blocked: 'Answer blocked',
            rerank_blocked: 'Rerank blocked',
            max_steps_forced: 'Max steps forced',
            stop: 'Stop',
        };
        const label = labelMap[e.event] || e.event;
        const metaParts: string[] = [];
        if (typeof e.step === 'number') metaParts.push(`step ${e.step}${e.max_steps ? `/${e.max_steps}` : ''}`);
        if (typeof e.docs === 'number') metaParts.push(`docs ${e.docs}`);
        if (typeof e.top_k === 'number') metaParts.push(`k ${e.top_k}`);
        if (Array.isArray(e.tool_calls) && e.tool_calls.length > 0) metaParts.push(`tools ${e.tool_calls.join(', ')}`);
        if (typeof e.latency_s === 'number') metaParts.push(`${e.latency_s.toFixed(2)}s`);
        const meta = metaParts.join(' · ');
        const detail = e.query || '';
        return { label, meta, detail };
    };

    useEffect(() => {
        const loadHistory = async () => {
            setLoading(true);
            try {
                const response = await chatApi.getChatHistory(fileName);
                const historyMessages: ChatMessage[] = [];

                // History is returned descending (newest first), we want ascending for chat
                [...response.history].reverse().forEach((item: any) => {
                    // Current history item contains both Q and A
                    // We split them into two messages for the UI
                    historyMessages.push({
                        id: `${item.id}-q`,
                        sender: 'user',
                        text: item.question
                    });
                    historyMessages.push({
                        id: `${item.id}-a`,
                        sender: 'ai',
                        text: item.answer,
                        citations: item.citations
                    });
                });

                setMessages(historyMessages);
            } catch (error) {
                console.error('Failed to load chat history:', error);
                setMessages([]);
            } finally {
                setLoading(false);
            }
        };

        if (fileName) {
            loadHistory();
        }
    }, [fileName]);

    useEffect(() => {
        if (!agentMode) {
            setAgentActivity([]);
            setActivityOpen(false);
        }
    }, [agentMode]);
    const [streaming, setStreaming] = useState(false);
    const bufferRef = useRef<string>('');
    const revealedTextRef = useRef<string>('');
    const timerRef = useRef<any>(null);

    // Cleanup timer on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading || streaming) return;

        const userText = input.trim();
        const messageId = Date.now().toString();
        const aiId = messageId + '-ai';

        setInput('');
        setLoading(true);
        bufferRef.current = '';
        revealedTextRef.current = '';
        setAgentActivity([]);
        setActivityOpen(false);

        setMessages(prev => [...prev, { id: messageId, sender: 'user', text: userText }]);

        try {
            const params = new URLSearchParams({
                message: userText,
                file_name: fileName,
                use_reranker: 'true',
            });

            const streamPath = agentMode ? '/api/chat/agent/stream' : '/api/chat/stream';

            const response = await fetch(`${streamPath}?${params}`, {
                method: 'GET',
                credentials: 'include',
                headers: { Accept: 'text/event-stream' },
            });

            if (!response.ok) {
                setLoading(false);
                const err = await response.json().catch(() => ({ detail: 'Stream failed' }));
                setMessages(prev => [...prev, { id: aiId, sender: 'ai', text: `Error: ${err.detail || 'Failed to get response'}` }]);
                return;
            }

            const reader = response.body!.getReader();
            const decoder = new TextDecoder();
            let networkBuffer = '';
            let startedStreaming = false;

            // Start the typewriter timer
            timerRef.current = setInterval(() => {
                if (bufferRef.current.length > 0) {
                    // Adaptive speed: reveal more characters if buffer is large to keep up
                    // Very aggressive for "faster" feel: base 5, jumps to 25 if backlog is heavy
                    const revealCount = bufferRef.current.length > 40 ? 25 : (bufferRef.current.length > 15 ? 12 : 5);
                    const chunk = bufferRef.current.slice(0, revealCount);
                    bufferRef.current = bufferRef.current.slice(revealCount);
                    revealedTextRef.current += chunk;

                    setMessages(prev => prev.map(m =>
                        m.id === aiId ? { ...m, text: revealedTextRef.current } : m
                    ));
                }
            }, 15);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                networkBuffer += decoder.decode(value, { stream: true });
                const lines = networkBuffer.split('\n');
                networkBuffer = lines.pop() ?? '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const raw = line.slice(6).trim();
                    if (!raw) continue;

                    try {
                        const event = JSON.parse(raw);
                        if (event.type === 'token') {
                            if (!startedStreaming) {
                                startedStreaming = true;
                                setLoading(false);
                                setStreaming(true);
                                // Add the AI message once the first token arrives
                                setMessages(prev => [...prev, { id: aiId, sender: 'ai', text: '' }]);
                            }
                            bufferRef.current += event.text;
                        } else if (event.type === 'agent') {
                            if (agentMode) {
                                const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
                                setAgentActivity(prev => {
                                    const next = [...prev, {
                                        id,
                                        ts: Date.now(),
                                        event: event.event,
                                        step: event.step,
                                        max_steps: event.max_steps,
                                        query: event.query,
                                        previous_query: event.previous_query,
                                        docs: event.docs,
                                        top_k: event.top_k,
                                        reason: event.reason,
                                        tool_calls: event.tool_calls,
                                        latency_s: event.latency_s,
                                    } as AgentActivityItem];
                                    return next.slice(-30);
                                });
                            }
                        } else if (event.type === 'done') {
                            // Wait for buffer to clear before finishing
                            const waitFinish = setInterval(() => {
                                if (bufferRef.current.length === 0) {
                                    clearInterval(waitFinish);
                                    if (timerRef.current) clearInterval(timerRef.current);
                                    timerRef.current = null;
                                    setStreaming(false);
                                    setMessages(prev => prev.map(m =>
                                        m.id === aiId
                                            ? {
                                                  ...m,
                                                  text: typeof event.answer === 'string' ? event.answer : m.text,
                                                  citations: event.citations,
                                                  cached: event.cached,
                                              }
                                            : m
                                    ));
                                }
                            }, 100);
                        } else if (event.type === 'error') {
                            setLoading(false);
                            setStreaming(false);
                            if (timerRef.current) clearInterval(timerRef.current);
                            setMessages(prev => [...prev, { id: aiId, sender: 'ai', text: `Error: ${event.message}` }]);
                        }
                    } catch { /* ignore */ }
                }
            }
        } catch (error: any) {
            setLoading(false);
            setStreaming(false);
            if (timerRef.current) clearInterval(timerRef.current);
            setMessages(prev => [...prev, { id: aiId, sender: 'ai', text: `Error: ${error.message || 'Failed to get response'}` }]);
        } finally {
            // Main timer cleanup is handled in the 'done' handler or error catchers
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-header">
                <h3>{fileName || 'New Chat'}</h3>
                <div className="chat-header-right">
                    {agentMode && (
                        <div className="agent-activity-wrap">
                            <button
                                type="button"
                                className="agent-activity-btn"
                                onClick={() => setActivityOpen(v => !v)}
                                disabled={loading || streaming}
                            >
                                Activity
                                {agentActivity.length > 0 && (
                                    <span className="agent-activity-count">{agentActivity.length}</span>
                                )}
                            </button>
                            {activityOpen && agentActivity.length > 0 && (
                                <div className="agent-activity-popover">
                                    <div className="agent-activity-title">Agent Activity</div>
                                    <div className="agent-activity-list">
                                        {agentActivity.slice().reverse().slice(0, 12).map((a) => {
                                            const f = formatAgentEvent(a);
                                            return (
                                                <div key={a.id} className="agent-activity-item">
                                                    <div className="agent-activity-row">
                                                        <span className="agent-activity-badge">{f.label}</span>
                                                        {f.meta && <span className="agent-activity-meta">{f.meta}</span>}
                                                    </div>
                                                    {a.reason && <div className="agent-activity-reason">{a.reason}</div>}
                                                    {f.detail && <div className="agent-activity-query">{f.detail}</div>}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                    <label className="agent-toggle">
                        <input
                            type="checkbox"
                            checked={agentMode}
                            onChange={(e) => setAgentMode(e.target.checked)}
                            disabled={loading || streaming}
                        />
                        <span className="agent-toggle-ui" />
                        <span className="agent-toggle-text">Agent</span>
                    </label>
                </div>
            </div>

            <div className="chat-messages-container">
                <div className="chat-messages">
                    {messages.length === 0 ? (
                        <div className="chat-empty">
                            <div className="chat-empty-icon">
                                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                            </div>
                            <h2>How can I help you today?</h2>
                            <p>Ask questions about <strong>{fileName}</strong></p>
                        </div>
                    ) : (
                        messages.map((msg) => (
                            <div key={msg.id} className={`message-row ${msg.sender}`}>
                                <div className="message-content-wrapper">
                                    <div className="message-avatar">
                                        {msg.sender === 'user' ? (
                                            <div className="avatar-user">You</div>
                                        ) : (
                                            <div className="avatar-ai">
                                                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                                </svg>
                                            </div>
                                        )}
                                    </div>
                                    <div className="message-bubble">
                                        {msg.sender === 'ai' ? (
                                            <div className="markdown-content">
                                                <MarkdownRenderer
                                                    content={msg.text}
                                                    citations={msg.citations}
                                                />
                                                {msg.cached && <span className="cached-badge" title="Served from cache">⚡ Cached</span>}
                                                <div className="message-actions">
                                                    <button
                                                        className="action-btn"
                                                        onClick={() => navigator.clipboard.writeText(msg.text)}
                                                        title="Copy response"
                                                    >
                                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="16" height="16">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                                        </svg>
                                                    </button>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="user-text">{msg.text}</div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                    {loading && (
                        <div className="message-row ai">
                            <div className="message-content-wrapper">
                                <div className="message-avatar">
                                    <div className="avatar-ai">
                                        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                        </svg>
                                    </div>
                                </div>
                                <div className="message-bubble">
                                    {agentMode ? (
                                        <div className="agent-thinking">
                                            <div className="agent-thinking-header">
                                                <span className="agent-thinking-title">Agent Activity</span>
                                                {agentActivity.length > 0 && (
                                                    <span className="agent-thinking-count">{agentActivity.length}</span>
                                                )}
                                            </div>
                                            <div className="agent-thinking-list">
                                                {(agentActivity.length > 0 ? agentActivity.slice(-6) : [{ id: 'idle', ts: Date.now(), event: 'step_start' } as AgentActivityItem]).map((a) => {
                                                    const f = formatAgentEvent(a);
                                                    return (
                                                        <div key={a.id} className="agent-thinking-item">
                                                            <div className="agent-thinking-row">
                                                                <span className="agent-thinking-badge">{f.label}</span>
                                                                {f.meta && <span className="agent-thinking-meta">{f.meta}</span>}
                                                            </div>
                                                            {a.reason && <div className="agent-thinking-reason">{a.reason}</div>}
                                                            {f.detail && <div className="agent-thinking-detail">{f.detail}</div>}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="typing-indicator">
                                            <span className="thinking-text">Thinking</span>
                                            <span></span><span></span><span></span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} className="messages-end" />
                </div>
            </div>

            <div className="chat-input-container">
                <div className="chat-input-wrapper">
                    <form onSubmit={handleSubmit} className="chat-input-form">
                        <input
                            type="text"
                            className="chat-input"
                            placeholder="Message..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            className="send-btn"
                            disabled={loading || streaming || !input.trim()}
                        >
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                            </svg>
                        </button>
                    </form>
                    <div className="chat-footer-text">
                        AI can make mistakes. Consider checking important information.
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;

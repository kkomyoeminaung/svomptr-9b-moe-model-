/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, BrainCircuit, Mic, MicOff, Settings, Link, Loader2, Database } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import KnowledgeIngestion from './components/KnowledgeIngestion';
import SettingsModal from './components/SettingsModal';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  frame?: any;
  routing?: {
    main_expert: string;
    active_domain: string;
    confidence: number;
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'management'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [recognitionLang, setRecognitionLang] = useState('en-US');
  const [backendMode, setBackendMode] = useState(() => window.localStorage.getItem('svomptr_backend_mode') || 'mock');
  const [colabUrl, setColabUrl] = useState(() => window.localStorage.getItem('svomptr_colab_url') || '');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showClearSuccess, setShowClearSuccess] = useState(false);
  const recognitionRef = useRef<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync to localstorage
  useEffect(() => { window.localStorage.setItem('svomptr_backend_mode', backendMode); }, [backendMode]);
  useEffect(() => { window.localStorage.setItem('svomptr_colab_url', colabUrl); }, [colabUrl]);

  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');

  // Simple safe ID generator
  const generateId = () => Date.now().toString(36) + Math.random().toString(36).substring(2);

  const checkBackend = async () => {
    try {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), 3000);
        
        let fetchUrl = '/api/health';
        if (backendMode === 'colab' && colabUrl) {
            fetchUrl = `/api/health?colabUrl=${encodeURIComponent(colabUrl)}`;
        }
        
        const resp = await fetch(fetchUrl, { 
            signal: controller.signal
        });
        clearTimeout(id);
        setBackendStatus(resp.ok ? 'online' : 'offline');
        return resp.ok;
    } catch (e) {
        setBackendStatus('offline');
        return false;
    }
  };

  useEffect(() => {
    let interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, [backendMode, colabUrl]);

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
        try {
            const r = await fetch('/api/history');
            if (r.ok && isMounted) {
                const data = await r.json();
                if (Array.isArray(data)) {
                    const mapped = data.map((m: any) => ({
                      id: m.id || generateId(),
                      role: m.sender === 'user' ? 'user' : 'assistant' as any,
                      content: m.text || '',
                      frame: m.frame
                    }));
                    setMessages(mapped);
                }
            }
        } catch (e) {
            console.warn("History sync bypassed");
        }
    };
    init();
    return () => { isMounted = false; };
  }, []);

  const deleteHistory = async () => {
    console.log("[Maintenance] Requesting history clear...");
    setIsLoading(true);
    try {
        const resp = await fetch('/api/history', { method: 'DELETE' });
        if (resp.ok) {
            console.log("[Maintenance] History cleared successfully.");
            setMessages([]);
            setShowClearSuccess(true);
            setTimeout(() => setShowClearSuccess(false), 3000);
            
            // Critical: Also clear any local storage or refs that might be lingering
            window.localStorage.removeItem('svomptr_last_msg');
            return true;
        } else {
            const errData = await resp.json().catch(() => ({}));
            console.error("[Maintenance] Server failed to clear history:", resp.status, errData);
            alert("Server failed to clear history. Error: " + (errData.error || resp.statusText));
            return false;
        }
    } catch (e) {
        console.error("[Maintenance] Network error while clearing history:", e);
        alert("Network error: Could not reach the server to clear history.");
        return false;
    } finally {
        setIsLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);
  
  const toggleMicrophone = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (!SpeechRecognition) {
        setMessages(prev => [...prev, { 
          id: generateId(), 
          role: 'assistant', 
          content: "⚠️ Speech recognition is not supported in this browser." 
        }]);
        return;
      }
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.lang = recognitionLang;
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(prev => prev + " " + transcript);
      };
      recognition.onend = () => setIsListening(false);
      recognitionRef.current = recognition;
      recognition.start();
      setIsListening(true);
    }
  };

  const sendMessage = async () => {
    if (isLoading || (!input.trim() && !file)) {
        console.log("[Chat] Send blocked: empty input or loading");
        return;
    }

    const userMsgContent = input;
    console.log(`[Chat] Sending message: "${userMsgContent.substring(0, 50)}..."`);
    const userMessage: Message = { id: generateId(), role: 'user', content: userMsgContent };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let fileContent = "";
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        fileContent = ` [File: ${data.filename}]`;
        setFile(null);
      }

      let assistantContent = '';
      let frameData = null;

      try {
          const response = await fetch('/api/chat', {
              method: 'POST',
              headers: { 
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ 
                  message: userMsgContent + fileContent,
                  colabUrl: backendMode === 'colab' ? colabUrl : 'mock'
              }),
          });
          
          if (!response.ok) {
              throw new Error(`API connection failed (Status: ${response.status})`);
          }
          
          const data = await response.json();
          assistantContent = data.response || 'No response';
          frameData = data.frame || null;
          const routingData = data.routing || null;
          
          setMessages(prev => [...prev, {
            id: generateId(),
            role: 'assistant',
            content: assistantContent,
            frame: frameData,
            routing: routingData
          }]);
      } catch (err) {
          throw err; 
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: generateId(),
        role: 'assistant',
        content: "⚠️ Connection to local SVOMPTR-9B core lost. Please check your backend status."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] text-slate-900 font-sans overflow-hidden">
      {/* Sidebar Navigation - Desktop */}
      <aside className="w-20 lg:w-64 bg-white border-r border-slate-200 flex flex-col hidden md:flex shrink-0">
        <div className="p-6 flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shrink-0 shadow-lg shadow-blue-200">
            <BrainCircuit className="w-6 h-6 text-white" />
          </div>
          <span className="font-black text-xl tracking-tighter text-slate-900 hidden lg:block">SVOMPTR<span className="text-blue-600">9B</span></span>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4">
          <button 
            onClick={() => setActiveTab('chat')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all ${activeTab === 'chat' ? 'bg-blue-50 text-blue-600 shadow-sm shadow-blue-100' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'}`}
          >
            <Bot className="w-5 h-5 transition-transform group-hover:scale-110" />
            <span className="font-bold text-sm hidden lg:block text-left">Expert Chat</span>
          </button>
          <button 
            onClick={() => setActiveTab('management')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all ${activeTab === 'management' ? 'bg-blue-50 text-blue-600 shadow-sm shadow-blue-100' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'}`}
          >
            <Settings className="w-5 h-5" />
            <span className="font-bold text-sm hidden lg:block text-left">Neural Ingestion</span>
          </button>
          
            <button 
              onClick={() => setIsSettingsOpen(true)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-slate-500 hover:bg-slate-50 hover:text-slate-900 transition-all font-bold text-sm"
            >
              <Settings className="w-5 h-5" />
              <span className="hidden lg:block">Neural Configuration</span>
            </button>

            <button 
              onClick={() => {
                const confirmed = window.confirm("PURGE NEURAL CONTEXT?\n\nThis will permanently delete all chat messages from the neural history database.");
                if (confirmed) {
                    deleteHistory();
                }
              }}
              disabled={isLoading}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-rose-400 hover:bg-rose-50 hover:text-rose-600 transition-all font-black text-sm relative z-30 disabled:opacity-50"
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Database className="w-5 h-5" />}
              <span className="hidden lg:block">Clear Neural Context</span>
            </button>

            <div className="pt-6 hidden lg:block">
             <p className="text-[10px] font-black uppercase text-slate-400 px-4 mb-2 tracking-widest">Inference Source</p>
             <div className="px-2 space-y-2">
                 <select 
                    className="w-full bg-slate-50 border border-slate-200 text-slate-600 text-xs font-bold rounded-xl p-2 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-100"
                    value={backendMode}
                    onChange={(e) => setBackendMode(e.target.value)}
                 >
                     <option value="mock">Python Mock Pipeline (Demo)</option>
                     <option value="colab">Colab Live Backend (GPU)</option>
                 </select>

                 {backendMode === 'colab' && (
                     <div className="space-y-1">
                         <input
                             type="text"
                             placeholder="Paste .loca.lt URL..."
                             className="w-full bg-white border border-slate-200 text-slate-800 text-[11px] font-bold rounded-xl p-2 outline-none focus:border-amber-400 focus:ring-2 focus:ring-amber-50"
                             value={colabUrl}
                             onChange={(e) => setColabUrl(e.target.value)}
                         />
                         {!colabUrl && (
                             <div className="flex items-center gap-1.5 px-2 py-1 bg-amber-50 rounded-lg border border-amber-100">
                                 <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                                 <p className="text-[9px] font-black text-amber-700 uppercase">Gateway URL Required</p>
                             </div>
                         )}
                     </div>
                 )}
             </div>
          </div>
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className={`flex items-center gap-3 px-4 py-3 rounded-2xl ${backendStatus === 'online' ? 'bg-emerald-50' : 'bg-rose-50'}`}>
            <div className={`w-2 h-2 rounded-full shrink-0 ${backendStatus === 'online' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            <div className="hidden lg:block overflow-hidden">
                <p className={`text-[10px] font-black uppercase tracking-wider truncate ${backendStatus === 'online' ? 'text-emerald-700' : 'text-rose-700'}`}>
                Core {backendStatus === 'online' ? 'Synchronized' : 'Offline'}
                </p>
                <p className="text-[10px] text-slate-400 font-medium">9.0.21-Stable</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative">
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center justify-between px-6 sticky top-0 z-20 md:hidden">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-6 h-6 text-blue-600" />
            <span className="font-black text-lg tracking-tighter">SVOMPTR</span>
          </div>
          <div className="flex gap-1">
            <button 
              onClick={() => {
                if (window.confirm("Purge history?")) deleteHistory();
              }} 
              disabled={isLoading}
              className="p-3 text-slate-400 hover:text-rose-500 active:scale-90 transition-all font-bold disabled:opacity-50"
              title="Clear History"
            >
               {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Database className="w-5 h-5 text-rose-400" />}
            </button>
            <button onClick={() => setActiveTab('chat')} className={`p-2 rounded-lg ${activeTab === 'chat' ? 'bg-blue-50 text-blue-600' : 'text-slate-400'}`}><Bot className="w-5 h-5" /></button>
            <button onClick={() => setIsSettingsOpen(true)} className="p-2 text-slate-400"><Settings className="w-5 h-5" /></button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto relative">
          {/* Subtle Background */}
          <div className="absolute inset-0 opacity-[0.02] pointer-events-none select-none z-0" style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '32px 32px' }} />
          
          <div className="relative z-10 h-full">
            {activeTab === 'chat' ? (
              <div className="h-full flex flex-col max-w-4xl mx-auto">
                <div className="flex-1 overflow-y-auto p-4 md:p-10 space-y-8 scroll-smooth">
                  {messages.length === 0 ? (
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="h-full flex flex-col items-center justify-center text-center space-y-8">
                      <div className="relative">
                          <div className="w-24 h-24 bg-white rounded-[2.5rem] shadow-2xl shadow-blue-100 flex items-center justify-center border border-slate-50">
                            <Bot className="w-12 h-12 text-blue-600" />
                          </div>
                          <motion.div 
                            animate={{ rotate: 360 }} 
                            transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                            className="absolute -inset-2 border-2 border-dashed border-blue-200 rounded-[3rem] opacity-50"
                          />
                      </div>
                      <div className="space-y-3">
                        <h1 className="text-4xl font-black tracking-tight text-slate-900">Neural Gateway Alpha</h1>
                        <p className="text-slate-400 font-medium max-w-sm mx-auto text-sm leading-relaxed">Welcome to the SVOMPTR architecture ecosystem. Analyze, distill, and interact with the next generation of grammatical AI.</p>
                      </div>
                      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
                        {['Explain SVOMPTR reasoning', 'Analyze this sentence structure', 'Historical context of AI', 'Scientific classification help'].map(hint => (
                          <button key={hint} onClick={() => setInput(hint)} className="px-5 py-2.5 bg-white border border-slate-200 rounded-2xl text-[11px] font-bold text-slate-500 hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50/30 transition-all shadow-sm">
                            {hint}
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  ) : (
                    messages.map((msg) => (
                      <motion.div 
                        key={msg.id}
                        initial={{ opacity: 0, y: 10, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        className={`flex gap-5 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                      >
                        <div className={`w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 shadow-lg ${msg.role === 'user' ? 'bg-slate-900' : 'bg-blue-600 shadow-blue-200'}`}>
                          {msg.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-6 h-6 text-white" />}
                        </div>
                        <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                          {msg.routing && (
                            <div className="w-full flex items-center gap-3 px-4 py-2 bg-slate-900/5 rounded-2xl border border-slate-100 mb-1">
                                <div className="flex items-center gap-2">
                                    <BrainCircuit className="w-3 h-3 text-indigo-500" />
                                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-tighter">Neural Gate:</span>
                                </div>
                                <div className="flex-1 flex items-center gap-2">
                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${msg.routing.active_domain !== 'general' ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-600'}`}>
                                        {msg.routing.active_domain.toUpperCase()} EXPERT
                                    </span>
                                    <div className="flex-1 h-1 bg-slate-200 rounded-full overflow-hidden max-w-[60px]">
                                        <motion.div 
                                            initial={{ width: 0 }}
                                            animate={{ width: `${msg.routing.confidence * 100}%` }}
                                            className="h-full bg-indigo-500"
                                        />
                                    </div>
                                    <span className="text-[9px] font-black text-indigo-500">{(msg.routing.confidence * 100).toFixed(0)}%</span>
                                </div>
                                {msg.routing.self_refined && (
                                    <motion.div 
                                      initial={{ opacity: 0, scale: 0.9 }}
                                      animate={{ opacity: 1, scale: 1 }}
                                      className="flex items-center gap-1.5 px-2 py-0.5 bg-rose-50 border border-rose-100 rounded-md"
                                      title={msg.routing.critique}
                                    >
                                        <div className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse" />
                                        <span className="text-[9px] font-black text-rose-600 uppercase">Self-Corrected</span>
                                    </motion.div>
                                )}
                            </div>
                          )}
                          <div className={`px-6 py-5 rounded-3xl shadow-sm leading-relaxed whitespace-pre-wrap text-[15px] ${
                            msg.role === 'user' 
                              ? 'bg-white border border-slate-200 text-slate-900 rounded-tr-none' 
                              : 'bg-white border border-blue-50 text-slate-700 rounded-tl-none ring-1 ring-blue-100/10'
                          }`}>
                            {msg.content}
                          </div>
                          {msg.frame && (
                            <div className="w-full bg-white/40 backdrop-blur-sm border border-slate-100 rounded-3xl p-5 mt-2">
                                <div className="flex items-center justify-between mb-4">
                                  <div className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse" />
                                    <span className="text-[10px] font-black text-blue-500 uppercase tracking-widest px-2 py-0.5 bg-blue-50 rounded-md">Neural Trace v9.0</span>
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-300 uppercase">Synchronous Routing</span>
                                </div>
                                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                                  {msg.frame && typeof msg.frame === 'object' && Object.entries(msg.frame).map(([k, v]) => (
                                    <div key={k} className="bg-slate-50/50 p-3 rounded-2xl border border-slate-100/50">
                                      <p className="text-[9px] font-bold text-slate-400 uppercase mb-1 tracking-tighter">{k}</p>
                                      <p className="text-[11px] font-black text-slate-800 truncate" title={String(v)}>{String(v) || '-'}</p>
                                    </div>
                                  ))}
                                </div>
                            </div>
                          )}
                          {msg.role === 'assistant' && (
                            <div className="flex gap-2 mt-2">
                                <button className="px-3 py-1 bg-white border border-slate-100 rounded-lg text-[10px] font-bold text-slate-400 hover:text-blue-600 hover:border-blue-200 transition-all">
                                    Helpful
                                </button>
                                <button 
                                    onClick={() => {
                                        const correct = window.prompt("Teach me: What should be the correct translation/structure?");
                                        if (correct) {
                                            setInput(`Correction: For "${msg.content.split('\n')[0].replace('Translation:', '').strip()}", it should be "${correct}"`);
                                            sendMessage();
                                        }
                                    }}
                                    className="px-3 py-1 bg-white border border-slate-100 rounded-lg text-[10px] font-bold text-slate-400 hover:text-rose-600 hover:border-rose-200 transition-all"
                                >
                                    Improve Answer
                                </button>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    ))
                  )}
                  {isLoading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-5">
                      <div className="w-10 h-10 rounded-2xl bg-blue-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-200 animate-pulse">
                        <Bot className="w-6 h-6 text-white" />
                      </div>
                      <div className="px-6 py-5 bg-white border border-blue-50 rounded-3xl rounded-tl-none shadow-sm flex items-center gap-3">
                        <div className="flex gap-1.5">
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" />
                        </div>
                        <span className="text-xs font-bold text-blue-400 uppercase tracking-wider">Expert Routing...</span>
                      </div>
                    </motion.div>
                  )}
                  <div ref={messagesEndRef} className="h-4" />
                </div>

                {/* Better Input Area */}
                <div className="p-6 md:p-10 bg-gradient-to-t from-[#F8FAFC] via-[#F8FAFC] to-transparent">
                  <div className="max-w-4xl mx-auto relative group">
                    <div className="absolute -inset-1 bg-gradient-to-r from-blue-400 to-indigo-400 rounded-[3rem] opacity-0 group-focus-within:opacity-10 transition-opacity blur" />
                    <form 
                      onSubmit={(e) => { e.preventDefault(); sendMessage(); }}
                      className="relative bg-white border border-slate-200 focus-within:border-blue-500 rounded-[2.5rem] p-2 flex items-center shadow-2xl shadow-slate-200/40 transition-all"
                    >
                      <div className="flex items-center gap-1 pl-2">
                        <label className="p-3 text-slate-400 hover:text-blue-600 transition-colors cursor-pointer rounded-full hover:bg-slate-50">
                          <input type="file" className="hidden" onChange={e => setFile(e.target.files?.[0] || null)} />
                          <Link className="w-5 h-5" />
                        </label>
                        <button 
                            type="button"
                            onClick={toggleMicrophone}
                            className={`p-3 rounded-full transition-all ${isListening ? 'bg-red-50 text-red-600' : 'text-slate-400 hover:text-blue-600 hover:bg-slate-50'}`}
                        >
                            {isListening ? <MicOff className="w-5 h-5 animate-pulse" /> : <Mic className="w-5 h-5" />}
                        </button>
                      </div>
                      
                      <div className="flex-1 flex flex-col px-4 min-w-0">
                        {file && (
                            <div className="flex items-center gap-2 mb-1">
                                <span className="px-2 py-0.5 bg-blue-50 text-blue-600 text-[10px] font-bold rounded-md flex items-center gap-1 border border-blue-100">
                                    <Link className="w-2 h-2" /> {file.name}
                                    <button onClick={() => setFile(null)} className="ml-1 hover:text-red-500">×</button>
                                </span>
                            </div>
                        )}
                        <input 
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                            placeholder="Type a research query or structural analysis request..."
                            className="w-full py-3 bg-transparent border-none focus:ring-0 text-sm font-medium placeholder:text-slate-300"
                        />
                      </div>

                      <button 
                        type="submit"
                        disabled={(!input.trim() && !file) || isLoading}
                        className="bg-blue-600 text-white w-12 h-12 rounded-full flex items-center justify-center hover:bg-blue-700 disabled:opacity-50 shadow-lg shadow-blue-200 transition-all active:scale-95 shrink-0"
                      >
                        {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                      </button>
                    </form>
                    <div className="flex justify-center gap-6 mt-4">
                        <p className="text-[9px] text-slate-300 font-black uppercase tracking-[0.2em] pointer-events-none">SVOMPTR v9.2 • MoE-Optimized Core</p>
                        <div className="flex gap-2">
                           <button onClick={() => setRecognitionLang('en-US')} className={`text-[9px] font-black tracking-widest ${recognitionLang === 'en-US' ? 'text-blue-500' : 'text-slate-300'}`}>EN</button>
                           <button onClick={() => setRecognitionLang('my-MM')} className={`text-[9px] font-black tracking-widest ${recognitionLang === 'my-MM' ? 'text-blue-500' : 'text-slate-300'}`}>MY</button>
                        </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full overflow-y-auto">
                <KnowledgeIngestion />
              </div>
            )}
          </div>
        </main>
      </div>

      <SettingsModal 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        backendMode={backendMode}
        setBackendMode={setBackendMode}
        colabUrl={colabUrl}
        setColabUrl={setColabUrl}
        backendStatus={backendStatus}
        onClearHistory={deleteHistory}
        isLoading={isLoading}
      />

      <AnimatePresence>
        {showClearSuccess && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[100] px-6 py-3 bg-slate-900 text-white rounded-full shadow-2xl flex items-center gap-3 border border-slate-800"
          >
            <Database className="w-4 h-4 text-emerald-400" />
            <span className="text-[11px] font-black uppercase tracking-widest">Neural Context Purged</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

import { useState } from 'react';
import { Upload, Link, Loader2, BrainCircuit, Download, Settings, Database } from 'lucide-react';
import { motion } from 'framer-motion';
import DatasetDownloader from './DatasetDownloader';

export default function KnowledgeIngestion() {
  const [file, setFile] = useState<File | null>(null);
  const [subjects, setSubjects] = useState('');
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [generationText, setGenerationText] = useState('');
  const [grammarRules, setGrammarRules] = useState<string[]>([]);
  const [showRules, setShowRules] = useState(false);

  const fetchRules = async () => {
    setLoading(true);
    try {
      const resp = await fetch('/api/grammar-rules');
      if (resp.ok) {
        const data = await resp.json();
        setGrammarRules(data.rules || []);
        setShowRules(true);
      }
    } catch { setStatus('❌ Failed to fetch rules'); }
    setLoading(false);
  };

  const handleSynthesize = async () => {
    setLoading(true);
    setStatus('🧠 Initiating Recursive Neural Synthesis...');
    try {
      const resp = await fetch('/api/synthesize', { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setStatus(`✅ Synthesis Complete: Extracted ${data.new_rules_count} potential rules.`);
        fetchRules();
      }
    } catch { setStatus('❌ Neural Synthesis failed'); }
    setLoading(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    const backendMode = window.localStorage.getItem('svomptr_backend_mode') || 'mock';
    const colabUrl = window.localStorage.getItem('svomptr_colab_url') || '';
    if (backendMode === 'colab' && colabUrl) {
      formData.append('colabUrl', colabUrl);
    }
    
    try {
      const resp = await fetch('/api/upload', { method: 'POST', body: formData });
      if (resp.ok) setStatus('✅ File ingested into neural vault');
      else setStatus('❌ Upload failed');
    } catch { setStatus('❌ Connection error'); }
    setLoading(false);
  };

  const handleStartLearning = async () => {
    setLoading(true);
    const backendMode = window.localStorage.getItem('svomptr_backend_mode') || 'mock';
    const colabUrl = window.localStorage.getItem('svomptr_colab_url') || '';

    try {
      const resp = await fetch('/api/start-learning', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            colabUrl: backendMode === 'colab' ? colabUrl : 'mock'
        })
      });
      if (resp.ok) {
          const data = await resp.json();
          setStatus(`✅ ${data.message || 'Neural learning process initialized'}`);
      } else {
          setStatus('❌ Initialization failed');
      }
    } catch { setStatus('❌ Connection error'); }
    setLoading(false);
  };

  const handleUpdateSubjects = async () => {
    const subjectList = subjects.split(',').map(s => s.trim());
    setLoading(true);
    try {
      const resp = await fetch('/api/subjects', { 
         method: 'POST', 
         headers: {'Content-Type': 'application/json'},
         body: JSON.stringify({ subjects: subjectList }) 
      });
      if (resp.ok) setStatus('✅ Subject alignment updated');
      else setStatus('❌ Update failed');
    } catch { setStatus('❌ Connection error'); }
    setLoading(false);
  };

  const handlePackage = async () => {
    setLoading(true);
    setStatus('📦 Packaging neural adapter...');
    try {
      const resp = await fetch('/api/package-project');
      if (resp.ok) {
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'svomptr_adapter.zip';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setStatus('✅ Package exported successfully');
      } else {
        setStatus('❌ Packaging failed');
      }
    } catch { setStatus('❌ Packaging failed'); }
    setLoading(false);
  };

  const handleGenerate = async (format: string) => {
    setLoading(true);
    try {
      const resp = await fetch('/api/generate-file', { 
         method: 'POST', 
         headers: {'Content-Type': 'application/json'},
         body: JSON.stringify({ format, text: generationText }) 
      });

      if (resp.ok) {
        const contentType = resp.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
           const data = await resp.json();
           setStatus(data.message || '✅ Manifest generated');
        } else {
           const blob = await resp.blob();
           const url = window.URL.createObjectURL(blob);
           const a = document.createElement('a');
           a.href = url;
           a.download = `neural_report.${format}`;
           document.body.appendChild(a);
           a.click();
           a.remove();
           setStatus(`✅ PDF manifest generated.`);
        }
      } else {
        setStatus('❌ Generation failed');
      }
    } catch { setStatus('❌ Connection error'); }
    setLoading(false);
  };

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 space-y-10 pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-50 border border-blue-100 rounded-full text-[10px] font-black text-blue-600 uppercase tracking-widest">
                <Settings className="w-3 h-3" /> System Configuration
            </div>
            <h2 className="text-4xl font-black tracking-tight text-slate-900">Neural Ingestion</h2>
            <p className="text-slate-500 font-medium max-w-lg">Fine-tune the SVOMPTR-9B core by injecting specialized knowledge domains and configuring neural routing parameters.</p>
        </div>

        <div className="flex gap-4">
            <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm min-w-[120px]">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Total Weights</p>
                <p className="text-xl font-black text-slate-900">9.1B</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm min-w-[120px]">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Inference Speed</p>
                <p className="text-xl font-black text-emerald-600">42 t/s</p>
            </div>
        </div>
      </div>
      
      {/* Grammar Rules Transparency Section */}
      <section className="bg-white p-8 rounded-[2.5rem] border border-blue-100 shadow-sm space-y-6">
          <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-2xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-200 text-white">
                      <Database className="w-6 h-6" />
                  </div>
                  <div>
                      <h3 className="text-xl font-black text-slate-900">Grammar Rules Repository</h3>
                      <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Structural Memory Management</p>
                  </div>
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={handleSynthesize}
                  disabled={loading}
                  className="px-6 py-2 bg-rose-50 text-rose-600 rounded-xl font-black text-xs hover:bg-rose-100 transition-all border border-rose-100 flex items-center gap-2"
                >
                    {loading && <Loader2 className="w-3 h-3 animate-spin" />}
                    Autonomous Learning
                </button>
                <button 
                  onClick={fetchRules}
                  className="px-6 py-2 bg-indigo-50 text-indigo-600 rounded-xl font-black text-xs hover:bg-indigo-100 transition-all"
                >
                    {showRules ? 'Refresh Repository' : 'View Core Rules'}
                </button>
              </div>
          </div>
          
          {showRules && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                  {grammarRules.length > 0 ? (
                      grammarRules.map((rule, idx) => (
                          <div key={idx} className="p-4 bg-slate-50 border border-slate-100 rounded-2xl">
                              <p className="text-[10px] font-black text-indigo-400 mb-2 uppercase">Rule #{idx + 1}</p>
                              <code className="text-[11px] font-mono text-slate-700 block bg-white p-3 rounded-lg border border-slate-200">
                                  {rule}
                              </code>
                          </div>
                      ))
                  ) : (
                      <div className="col-span-2 py-10 text-center border-2 border-dashed border-slate-100 rounded-3xl">
                          <p className="text-slate-400 font-bold text-sm italic">Neural memory is currently pristine. Learn from chat interactions to populate.</p>
                      </div>
                  )}
              </div>
          )}
      </section>

      {/* System Health & Architecture Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-indigo-600 rounded-[2rem] text-white shadow-xl shadow-indigo-100 flex flex-col justify-between">
              <div>
                  <h4 className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Compute Core</h4>
                  <p className="text-2xl font-black">SVOMPTR-9B</p>
              </div>
              <div className="mt-4 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-[11px] font-bold">Neural Engine: Active</span>
              </div>
          </div>
          
          <div className="p-6 bg-white border border-slate-200 rounded-[2rem] shadow-sm">
              <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Mixture-of-Experts</h4>
              <p className="text-2xl font-black text-slate-900">8 Experts</p>
              <p className="text-[11px] font-bold text-indigo-500 mt-2">Active Routing: Dynamic</p>
          </div>

          <div className="p-6 bg-white border border-slate-200 rounded-[2rem] shadow-sm">
              <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Knowledge Density</h4>
              <p className="text-2xl font-black text-slate-900">{grammarRules.length + 142}k</p>
              <p className="text-[11px] font-bold text-rose-500 mt-2">RAG Context: Optimized</p>
          </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section className="bg-white p-8 rounded-[2.5rem] border border-slate-200/60 shadow-sm space-y-8">
          <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-200 font-bold text-white uppercase text-xs">
                  <Upload className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-black text-slate-900">Knowledge Vault</h3>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Context Injection</p>
              </div>
          </div>
          
          <div className="space-y-4">
              <p className="text-sm text-slate-500 leading-relaxed font-medium">Feed the neural engine with specialized literature. Supports multi-encoding for PDF, DOCX, and structural Markdown.</p>
              
              <div className="space-y-3">
                <label className="group relative block w-full border-2 border-dashed border-slate-200 rounded-3xl p-6 transition-all hover:border-blue-400 hover:bg-blue-50/50 cursor-pointer text-center">
                    <input 
                        type="file" 
                        onChange={e => setFile(e.target.files?.[0] || null)} 
                        className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    <Upload className="w-8 h-8 text-slate-300 mx-auto mb-2 group-hover:text-blue-500 transition-colors" />
                    <p className="text-sm font-bold text-slate-600">{file ? file.name : "Drop context file here or click to browse"}</p>
                    <p className="text-[10px] font-bold text-slate-400 mt-1 uppercase">Max 50MB per neural slice</p>
                </label>

                <button 
                    onClick={handleUpload} 
                    disabled={loading || !file} 
                    className="w-full h-14 font-black bg-blue-600 text-white rounded-2xl hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-3 transition-all active:scale-95 shadow-xl shadow-blue-200/50"
                >
                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Upload className="w-5 h-5" />}
                    Ingest into Core
                </button>
              </div>
          </div>
        </section>
        
        <section className="bg-white p-8 rounded-[2.5rem] border border-slate-200/60 shadow-sm space-y-8">
          <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-200 font-bold text-white">
                  <BrainCircuit className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-black text-slate-900">Neural Focus & Auto-Train</h3>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Optimization Module</p>
              </div>
          </div>

          <div className="space-y-4">
              <p className="text-sm text-slate-500 leading-relaxed font-medium">Define parameters or trigger massive scale Auto-Training using Colab GPU capabilities on your distilled 1M dataset.</p>
              
              <div className="space-y-4">
                <div className="space-y-2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Target Domains</label>
                    <input 
                        type="text" 
                        value={subjects} 
                        onChange={e => setSubjects(e.target.value)} 
                        placeholder="e.g. Grammar, Syntax, Logical Reasoning..." 
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-6 py-4 text-sm font-bold placeholder:text-slate-300 focus:bg-white focus:ring-2 focus:ring-blue-500 outline-none transition-all" 
                    />
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <button 
                        onClick={handleUpdateSubjects} 
                        disabled={loading} 
                        className="h-14 font-black bg-slate-100 text-slate-600 rounded-2xl hover:bg-slate-200 transition-all active:scale-95"
                    >
                        Sync Subject
                    </button>
                    <a 
                        href="/api/download-colab" 
                        target="_blank"
                        className="h-14 font-black bg-amber-500 text-white rounded-2xl hover:bg-amber-600 transition-all active:scale-95 shadow-xl shadow-amber-200 flex items-center justify-center text-center leading-none"
                    >
                        Get Auto-Train
                    </a>
                </div>
                <a 
                    href="/api/download-distillation" 
                    target="_blank"
                    className="w-full h-10 font-bold bg-purple-100 text-purple-700 rounded-xl hover:bg-purple-200 transition-all active:scale-95 flex items-center justify-center text-xs"
                >
                    Download Phase-1 (Distillation)
                </a>
                <a 
                    href="/api/download-inference" 
                    target="_blank"
                    className="w-full h-10 font-bold bg-emerald-100 text-emerald-700 rounded-xl hover:bg-emerald-200 transition-all active:scale-95 flex items-center justify-center text-xs"
                >
                    Download Phase-3 (Live Inference Server)
                </a>
                <DatasetDownloader />
              </div>
          </div>
        </section>
      </div>

      <section className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl shadow-blue-900/10 relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-600 rounded-full blur-[120px] opacity-20 -mr-20 -mt-20 pointer-events-none group-hover:opacity-40 transition-opacity" />
        
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-10">
            <div className="space-y-4 max-w-xl">
                <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center border border-white/10">
                        <Download className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                        <h3 className="text-2xl font-black">Export weights</h3>
                        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Model Serialization</p>
                    </div>
                </div>
                <p className="text-slate-400 font-medium text-sm leading-relaxed">Package your custom-trained SVOMPTR adapter into a production-ready manifest for deployment in high-concurrency environments.</p>
                
                <div className="flex flex-wrap gap-2 pt-2">
                    {['v9.2-Stable', 'Quantized', 'PDF Manifest', 'Adapter Core'].map(tag => (
                        <span key={tag} className="px-3 py-1 rounded-full bg-white/5 border border-white/5 text-[9px] font-black uppercase tracking-widest text-slate-300">
                            {tag}
                        </span>
                    ))}
                </div>
            </div>

            <div className="w-full lg:w-80 space-y-3">
                <input 
                    type="text" 
                    value={generationText} 
                    onChange={e => setGenerationText(e.target.value)} 
                    placeholder="Project tag or manifest name..." 
                    className="w-full bg-white/5 border border-white/10 rounded-2xl px-6 py-4 text-sm font-black focus:ring-2 focus:ring-blue-500 outline-none transition-all placeholder:text-slate-600" 
                />
                <button 
                    onClick={() => handleGenerate('pdf')} 
                    disabled={loading} 
                    className="w-full h-14 font-black bg-blue-600 text-white rounded-2xl hover:bg-blue-500 transition-all active:scale-95 flex items-center justify-center gap-3 shadow-lg shadow-blue-900/50"
                >
                    Generate PDF Report
                </button>
                <button 
                    onClick={handlePackage} 
                    disabled={loading} 
                    className="w-full h-14 font-black bg-white text-slate-900 rounded-2xl hover:bg-slate-100 transition-all active:scale-95 flex items-center justify-center gap-3"
                >
                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                    Download Package
                </button>
            </div>
        </div>
      </section>

      {status && (
        <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-10 left-1/2 -translate-x-1/2 z-50 px-8 py-4 bg-white border border-slate-200 rounded-full shadow-2xl flex items-center gap-4 min-w-[300px]"
        >
            <div className="w-2.5 h-2.5 bg-blue-600 rounded-full animate-pulse" />
            <p className="text-sm font-black text-slate-900 tracking-tight">{status}</p>
            <button onClick={() => setStatus(null)} className="ml-auto text-slate-400 hover:text-slate-900 text-lg font-black leading-none">×</button>
        </motion.div>
      )}
    </div>
  );
}

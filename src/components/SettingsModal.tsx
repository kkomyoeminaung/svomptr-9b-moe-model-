import React from 'react';
import { X, Globe, Cpu, Link as LinkIcon, Database } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  backendMode: string;
  setBackendMode: (mode: string) => void;
  colabUrl: string;
  setColabUrl: (url: string) => void;
  backendStatus: string;
  onClearHistory: () => Promise<boolean>;
  isLoading: boolean;
}

export function SettingsModal({
  isOpen,
  onClose,
  backendMode,
  setBackendMode,
  colabUrl,
  setColabUrl,
  backendStatus,
  onClearHistory,
  isLoading
}: SettingsModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-md bg-white rounded-[2rem] shadow-2xl overflow-hidden border border-slate-100"
          >
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-50 text-blue-600 rounded-xl">
                  <Database className="w-5 h-5" />
                </div>
                <h2 className="font-black text-lg tracking-tight">Neural Configuration</h2>
              </div>
              <button onClick={onClose} className="p-2 text-slate-400 hover:text-slate-900 hover:bg-slate-50 rounded-full transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              <div className="space-y-3">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-400 px-1">Inference Engine</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setBackendMode('mock')}
                    className={`flex flex-col items-center gap-3 p-4 rounded-2xl border-2 transition-all ${
                      backendMode === 'mock' 
                        ? 'border-blue-600 bg-blue-50 text-blue-600' 
                        : 'border-slate-100 hover:border-slate-200 text-slate-500'
                    }`}
                  >
                    <Cpu className="w-6 h-6" />
                    <span className="text-xs font-bold">Local Mock</span>
                  </button>
                  <button
                    onClick={() => setBackendMode('colab')}
                    className={`flex flex-col items-center gap-3 p-4 rounded-2xl border-2 transition-all ${
                      backendMode === 'colab' 
                        ? 'border-blue-600 bg-blue-50 text-blue-600' 
                        : 'border-slate-100 hover:border-slate-200 text-slate-500'
                    }`}
                  >
                    <Globe className="w-6 h-6" />
                    <span className="text-xs font-bold">Colab Live</span>
                  </button>
                </div>
              </div>

              <AnimatePresence>
                {backendMode === 'colab' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-3 overflow-hidden"
                  >
                    <div className="flex items-center justify-between px-1">
                      <label className="text-[10px] font-black uppercase tracking-widest text-slate-400">LocalTunnel Gateway URL</label>
                      <button 
                        onClick={() => alert("Enter your .loca.lt link here. Do NOT set this in AI Studio Secrets/Environment unless you are an advanced user.")}
                        className="text-[9px] font-bold text-blue-500 hover:underline"
                      >
                        Help?
                      </button>
                    </div>
                    <div className="relative">
                      <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400">
                        <LinkIcon className="w-4 h-4" />
                      </div>
                      <input
                        type="text"
                        placeholder="e.g. svogateway9b.loca.lt"
                        value={colabUrl}
                        onChange={(e) => setColabUrl(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3 pl-11 pr-4 text-sm font-bold text-slate-900 focus:bg-white focus:border-blue-500 focus:ring-4 focus:ring-blue-50 outline-none transition-all placeholder:text-slate-300"
                      />
                    </div>
                    <p className="text-[10px] text-slate-400 italic px-1 leading-relaxed">
                      Copy the "your url is" output from Colab. This is the bridge between AI Studio and your personal GPU.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="space-y-3 pt-2">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-400 px-1">Maintenance</label>
                <button
                  onClick={async () => {
                    if (window.confirm("Are you sure you want to clear all neural history? This cannot be undone.")) {
                        const success = await onClearHistory();
                        if (success) {
                            onClose();
                        }
                    }
                  }}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 p-3 text-xs font-bold text-rose-600 bg-rose-50 border border-rose-100 rounded-xl hover:bg-rose-100 transition-all active:scale-[0.98] disabled:opacity-50"
                >
                  <Database className="w-4 h-4" />
                  {isLoading ? 'Processing Purge...' : 'Clear Neural Context (History)'}
                </button>
              </div>

              <div className={`p-4 rounded-2xl flex items-center justify-between ${backendStatus === 'online' ? 'bg-emerald-50 border border-emerald-100' : 'bg-rose-50 border border-rose-100'}`}>
                <div className="flex items-center gap-3">
                  <div className={`w-2.5 h-2.5 rounded-full ${backendStatus === 'online' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
                  <div>
                    <p className={`text-xs font-black uppercase tracking-wide ${backendStatus === 'online' ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {backendStatus === 'online' ? 'Synchronized' : 'Link Offline'}
                    </p>
                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-tighter">Latency: {backendStatus === 'online' ? '~42ms' : 'N/A'}</p>
                  </div>
                </div>
                <div className="text-[10px] font-black text-slate-300 bg-white/50 px-2 py-1 rounded-lg border border-slate-100">
                  V9.2.1-STABLE
                </div>
              </div>
            </div>

            <div className="p-6 bg-slate-50 border-t border-slate-100">
              <button
                onClick={onClose}
                className="w-full bg-slate-900 text-white font-black text-sm uppercase tracking-widest py-4 rounded-2xl hover:bg-slate-800 transition-all shadow-xl shadow-slate-200 active:scale-[0.98]"
              >
                Accept Calibration
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

export default SettingsModal;

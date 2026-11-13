import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { SystemInfoModal } from './components/SystemInfoModal';
import { ImageUploader } from './components/ImageUploader';
import { ProcessingTimeline } from './components/ProcessingTimeline';
import { ResultsViewer } from './components/ResultsViewer';
import { HealthResponse, ProcessResponse, ProcessingStage } from './types';
import { api } from './services/api';
import { AlertCircle, BookOpen, Sparkles, CheckCircle2 } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [isSystemInfoOpen, setIsSystemInfoOpen] = useState<boolean>(false);

  const [processingStage, setProcessingStage] = useState<ProcessingStage>('idle');
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Poll health status with auto-reconnect
  useEffect(() => {
    let mounted = true;
    const fetchHealth = async () => {
      try {
        const h = await api.getHealth();
        if (mounted) setHealth(h);
      } catch (err) {
        if (mounted) setHealth(null);
      } finally {
        if (mounted) setHealthLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 3000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleProcessImage = async (file: File, topicHint: string) => {
    setErrorMsg(null);
    setProcessingStage('uploading');

    // Smooth UI progress simulation while the CPU inference runs
    const t1 = setTimeout(() => setProcessingStage('detecting'), 400);
    const t2 = setTimeout(() => setProcessingStage('cropping'), 900);
    const t3 = setTimeout(() => setProcessingStage('analyzing'), 1400);
    const t4 = setTimeout(() => setProcessingStage('formatting'), 1900);

    try {
      const res = await api.processImage(file, topicHint);
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      setProcessingStage('complete');
      setTimeout(() => {
        setResult(res);
      }, 500);
    } catch (err: any) {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      setProcessingStage('error');
      setErrorMsg(err.message || 'An unexpected error occurred during image processing.');
    }
  };

  const handleReset = () => {
    setResult(null);
    setProcessingStage('idle');
    setErrorMsg(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <Header
        health={health}
        healthLoading={healthLoading}
        onOpenSystemInfo={() => setIsSystemInfoOpen(true)}
      />

      {/* System Info Modal */}
      <SystemInfoModal
        isOpen={isSystemInfoOpen}
        onClose={() => setIsSystemInfoOpen(false)}
        health={health}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Error Notification */}
        {errorMsg && (
          <div className="mb-6 max-w-4xl mx-auto p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3 text-rose-300 text-xs sm:text-sm">
            <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <strong className="font-semibold block text-white">Pipeline Execution Error</strong>
              <p>{errorMsg}</p>
              <button
                onClick={() => setErrorMsg(null)}
                className="text-xs text-rose-400 underline mt-1 block"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Processing Progress Modal */}
        {processingStage !== 'idle' && processingStage !== 'error' && !result && (
          <div className="animate-overlay-fade fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
            <ProcessingTimeline stage={processingStage} />
          </div>
        )}

        {/* Dynamic View: Uploader vs Results */}
        {!result ? (
          <ImageUploader
            onProcess={handleProcessImage}
            isProcessing={processingStage !== 'idle' && processingStage !== 'error'}
          />
        ) : (
          <ResultsViewer result={result} onReset={handleReset} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-sky-400" />
            <span className="font-semibold text-slate-400">BoardMind AI</span>
            <span>— Educational Computer Vision & Intelligent Notes Platform</span>
          </div>
          <div className="flex items-center gap-4 text-slate-500">
            <span>YOLOv8n CPU Engine</span>
            <span>•</span>
            <a href="http://127.0.0.1:8001/docs" target="_blank" rel="noreferrer" className="text-sky-400 hover:underline">
              API Docs (/docs)
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;


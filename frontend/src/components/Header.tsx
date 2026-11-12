import React from 'react';
import { Sparkles, Cpu, CheckCircle2, AlertTriangle, Layers, BookOpen } from 'lucide-react';
import { HealthResponse } from '../types';

interface HeaderProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onOpenSystemInfo: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, healthLoading, onOpenSystemInfo }) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <BookOpen className="h-5 w-5 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-tight text-white">BoardMind</span>
              <span className="px-1.5 py-0.5 text-xs font-semibold uppercase tracking-wider bg-sky-500/20 text-sky-400 rounded border border-sky-500/30">
                AI
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">From Whiteboard Scribbles to Smart Notes</p>
          </div>
        </div>

        {/* Backend & Model Status */}
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenSystemInfo}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-800 border border-slate-700/80 text-xs font-medium text-slate-300 transition-all hover:border-slate-600"
            title="Inspect trained ML models and hardware execution"
          >
            <Cpu className="h-3.5 w-3.5 text-sky-400" />
            <span className="hidden md:inline">YOLOv8n CPU Engine</span>
            {healthLoading ? (
              <span className="h-2 w-2 rounded-full bg-amber-400 animate-ping" />
            ) : isHealthy ? (
              <span className="flex items-center gap-1 text-emerald-400 font-semibold">
                <span className="h-2 w-2 rounded-full bg-emerald-400 inline-block"></span>
                <span>Active</span>
              </span>
            ) : (
              <span className="flex items-center gap-1 text-rose-400">
                <AlertTriangle className="h-3 w-3" />
                <span>Offline</span>
              </span>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};


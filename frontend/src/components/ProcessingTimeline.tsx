import React from 'react';
import { Upload, Eye, Crop, Edit3, FileText, CheckCircle2, Loader2, Sparkles } from 'lucide-react';
import { ProcessingStage } from '../types';

interface ProcessingTimelineProps {
  stage: ProcessingStage;
}

const STAGES = [
  { id: 'uploading', label: 'Upload', icon: Upload },
  { id: 'detecting', label: 'Detect', icon: Eye },
  { id: 'cropping', label: 'Enhance', icon: Crop },
  { id: 'analyzing', label: 'Read Text', icon: Edit3 },
  { id: 'formatting', label: 'Notes', icon: FileText },
];

export const ProcessingTimeline: React.FC<ProcessingTimelineProps> = ({ stage }) => {
  if (stage === 'idle' || stage === 'error') return null;

  const getStageIndex = (s: ProcessingStage): number => {
    switch (s) {
      case 'uploading': return 0;
      case 'detecting': return 1;
      case 'cropping': return 2;
      case 'analyzing': return 3;
      case 'formatting': return 4;
      case 'complete': return 5;
      default: return 0;
    }
  };

  const currentIndex = Math.min(getStageIndex(stage), STAGES.length - 1);
  const isComplete = stage === 'complete';
  const progress = isComplete ? 100 : ((currentIndex + 1) / STAGES.length) * 100;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-live="polite"
      aria-label="BoardMind AI processing pipeline"
      className="animate-modal-pop relative w-full max-w-lg overflow-hidden rounded-3xl border border-slate-700/70 bg-slate-900/95 p-6 shadow-2xl shadow-sky-950/60 sm:p-8"
    >
      {/* Soft glow accent */}
      <div className="pointer-events-none absolute -top-28 left-1/2 h-52 w-52 -translate-x-1/2 rounded-full bg-sky-500/20 blur-3xl" />

      {/* Spinner + title */}
      <div className="relative flex flex-col items-center text-center">
        <div className="relative mb-4 flex h-16 w-16 items-center justify-center">
          <span className="absolute inset-0 rounded-full border-2 border-sky-500/15" />
          <span
            className={`absolute inset-0 rounded-full border-2 border-transparent ${
              isComplete
                ? 'border-t-emerald-400 border-r-emerald-300'
                : 'border-t-sky-400 border-r-cyan-300 animate-spin'
            }`}
          />
          <span
            className={`flex h-11 w-11 items-center justify-center rounded-full ${
              isComplete ? 'bg-emerald-500 text-slate-950' : 'bg-sky-500/15 text-sky-300'
            }`}
          >
            {isComplete ? <CheckCircle2 className="h-6 w-6" /> : <Sparkles className="h-6 w-6" />}
          </span>
        </div>

        <h3 className="text-lg font-bold text-white">
          {isComplete ? 'Notes Generated!' : 'Analyzing your whiteboard'}
        </h3>
        <p className="mt-1 max-w-sm text-xs text-slate-400 sm:text-sm">
          {isComplete
            ? 'Preparing your structured study notes…'
            : 'Detecting the board, enhancing the image, and reading handwriting on CPU. This can take a few seconds.'}
        </p>
      </div>

      {/* Progress bar */}
      <div className="relative mt-6">
        <div className="mb-2 flex items-center justify-between text-[11px] font-medium text-slate-400">
          <span className="flex items-center gap-1.5">
            {!isComplete && <Loader2 className="h-3 w-3 animate-spin text-sky-400" />}
            {isComplete ? 'Pipeline complete' : STAGES[currentIndex].label}
          </span>
          <span className="font-mono tabular-nums">{Math.round(progress)}%</span>
        </div>
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
          {!isComplete && (
            <div className="animate-progress-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/50 to-transparent" />
          )}
        </div>
      </div>

      {/* Stage chips */}
      <div className="mt-6 grid grid-cols-5 gap-1.5 sm:gap-2">
        {STAGES.map((s, idx) => {
          const Icon = s.icon;
          const isDone = idx < currentIndex || isComplete;
          const isCurrent = idx === currentIndex && !isComplete;

          return (
            <div key={s.id} className="flex flex-col items-center">
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-full border-2 transition-all duration-300 sm:h-10 sm:w-10 ${
                  isDone
                    ? 'border-sky-400 bg-sky-500 text-slate-950 shadow-lg shadow-sky-500/30'
                    : isCurrent
                    ? 'border-sky-400 bg-slate-900 text-sky-400 ring-4 ring-sky-500/20'
                    : 'border-slate-700 bg-slate-900 text-slate-500'
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4 sm:h-5 sm:w-5" />
                ) : isCurrent ? (
                  <Loader2 className="h-4 w-4 animate-spin sm:h-5 sm:w-5" />
                ) : (
                  <Icon className="h-4 w-4" />
                )}
              </div>
              <span
                className={`mt-1.5 text-center text-[10px] font-medium leading-tight sm:text-xs ${
                  isDone || isCurrent ? 'text-sky-300' : 'text-slate-500'
                }`}
              >
                {s.label}
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-5 text-center text-[10px] text-slate-500 sm:text-[11px]">
        Five-stage pipeline · runs locally on your machine
      </p>
    </div>
  );
};

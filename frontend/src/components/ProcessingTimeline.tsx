import React from 'react';
import { Upload, Eye, Crop, Edit3, FileText, CheckCircle2, Loader2 } from 'lucide-react';
import { ProcessingStage } from '../types';

interface ProcessingTimelineProps {
  stage: ProcessingStage;
}

const STAGES = [
  { id: 'uploading', label: 'Upload Image', icon: Upload },
  { id: 'detecting', label: 'YOLO Detection', icon: Eye },
  { id: 'cropping', label: 'Crop & Contrast', icon: Crop },
  { id: 'analyzing', label: 'Stroke Analysis', icon: Edit3 },
  { id: 'formatting', label: 'Smart Notes', icon: FileText },
];

export const ProcessingTimeline: React.FC<ProcessingTimelineProps> = ({ stage }) => {
  if (stage === 'idle') return null;

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

  const currentIndex = getStageIndex(stage);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm max-w-4xl mx-auto my-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          {stage === 'complete' ? (
            <>
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Pipeline Complete! Notes Generated</span>
            </>
          ) : (
            <>
              <Loader2 className="h-4 w-4 text-sky-400 animate-spin" />
              <span>Executing BoardMind AI Multi-Stage Pipeline...</span>
            </>
          )}
        </h4>
        <span className="text-xs font-mono text-slate-400">
          Stage {Math.min(currentIndex + 1, 5)} of 5
        </span>
      </div>

      <div className="relative">
        {/* Progress Bar Line */}
        <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-800 -translate-y-1/2 z-0 rounded-full" />
        <div
          className="absolute top-1/2 left-0 h-1 bg-gradient-to-r from-sky-500 to-cyan-400 -translate-y-1/2 z-0 rounded-full transition-all duration-500"
          style={{ width: `${(currentIndex / 4) * 100}%` }}
        />

        {/* Steps */}
        <div className="relative z-10 flex justify-between items-center">
          {STAGES.map((s, idx) => {
            const Icon = s.icon;
            const isCompleted = idx < currentIndex || stage === 'complete';
            const isCurrent = idx === currentIndex && stage !== 'complete';

            return (
              <div key={s.id} className="flex flex-col items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                    isCompleted
                      ? 'bg-sky-500 border-sky-400 text-slate-950 shadow-lg shadow-sky-500/30'
                      : isCurrent
                      ? 'bg-slate-900 border-sky-400 text-sky-400 shadow-lg shadow-sky-500/20 ring-4 ring-sky-500/20 animate-pulse'
                      : 'bg-slate-900 border-slate-700 text-slate-500'
                  }`}
                >
                  {isCompleted ? (
                    <CheckCircle2 className="h-5 w-5 font-bold" />
                  ) : isCurrent ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                </div>
                <span
                  className={`text-xs mt-2 font-medium text-center transition-colors ${
                    isCompleted || isCurrent ? 'text-sky-300' : 'text-slate-500'
                  }`}
                >
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};


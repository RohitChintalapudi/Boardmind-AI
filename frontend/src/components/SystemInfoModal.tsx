import React from 'react';
import { X, Cpu, HardDrive, CheckCircle2, AlertCircle, Info, Database } from 'lucide-react';
import { HealthResponse } from '../types';

interface SystemInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthResponse | null;
}

export const SystemInfoModal: React.FC<SystemInfoModalProps> = ({ isOpen, onClose, health }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-xl w-full shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-800/40">
          <div className="flex items-center gap-2.5">
            <Cpu className="h-5 w-5 text-sky-400" />
            <h3 className="font-semibold text-white">BoardMind AI System & Model Architecture</h3>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-5 text-sm">
          {health ? (
            <>
              {/* Whiteboard Detector */}
              <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sky-400 font-semibold">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    <span>Whiteboard Object Detector (YOLOv8n)</span>
                  </div>
                  <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Loaded on {health.whiteboard_detector.device.toUpperCase()}
                  </span>
                </div>
                <div className="text-xs text-slate-400 font-mono break-all bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-500">Weights: </span>
                  {health.whiteboard_detector.weights_path}
                </div>
                <p className="text-xs text-slate-400">
                  Optimized for CPU edge inference with 3.14M parameters and 8.7 GFLOPs.
                </p>
              </div>

              {/* Handwriting Recognition */}
              <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-cyan-400 font-semibold">
                    <Info className="h-4 w-4 text-sky-400" />
                    <span>Handwriting Text Recognition (HTR)</span>
                  </div>
                  <span className="text-xs px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                    {health.handwriting_recognizer.status}
                  </span>
                </div>
                <div className="text-xs text-slate-400 space-y-1">
                  <p><span className="text-slate-500">Engine: </span>{health.handwriting_recognizer.engine}</p>
                  <p><span className="text-slate-500">Status: </span>{health.handwriting_recognizer.message}</p>
                </div>
              </div>

              {/* Hardware Execution */}
              <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60 space-y-2">
                <div className="flex items-center gap-2 text-slate-300 font-semibold">
                  <HardDrive className="h-4 w-4 text-slate-400" />
                  <span>Hardware & Environment</span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
                  <div className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80">
                    <span className="text-slate-500 block">Host Platform:</span>
                    {health.hardware.os} ({health.hardware.cpu_architecture})
                  </div>
                  <div className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80">
                    <span className="text-slate-500 block">Logical CPU Cores:</span>
                    {health.hardware.cpu_count} Threads
                  </div>
                  <div className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80">
                    <span className="text-slate-500 block">System RAM:</span>
                    {health.hardware.ram_total_gb} GB Total
                  </div>
                  <div className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80">
                    <span className="text-slate-500 block">RAM Utilization:</span>
                    {health.hardware.ram_used_percent}% in use
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-6 text-slate-400">
              <AlertCircle className="h-8 w-8 text-amber-400 mx-auto mb-2" />
              <p>Backend is not responding. Ensure FastAPI is running on port 8000.</p>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-slate-800/60 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};


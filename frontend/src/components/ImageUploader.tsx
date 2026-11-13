import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { Upload, Sparkles, FileQuestion, ArrowRight, AlertCircle } from 'lucide-react';

interface ImageUploaderProps {
  onProcess: (file: File, topicHint: string) => void;
  isProcessing: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onProcess, isProcessing }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topicHint, setTopicHint] = useState<string>('');
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setErrorMsg(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setErrorMsg('Please upload a valid image file (JPG, PNG, or WEBP).');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg('File size exceeds the 25MB limit.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsg('Please select or drop a whiteboard photo first.');
      return;
    }
    onProcess(selectedFile, topicHint);
  };

  // Quick sample loader
  const handleLoadSample = async () => {
    setErrorMsg(null);
    try {
      const sampleBlob = await fetch('/sample_whiteboard.jpg', { cache: 'no-cache' })
        .then(r => {
          if (!r.ok) throw new Error(`Sample image request failed: ${r.status}`);
          return r.blob();
        })
        .catch(async () => {
          const canvas = document.createElement('canvas');
          canvas.width = 800;
          canvas.height = 500;
          const ctx = canvas.getContext('2d')!;
          ctx.fillStyle = '#f8fafc';
          ctx.fillRect(0, 0, 800, 500);
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 14;
          ctx.strokeRect(20, 20, 760, 460);
          ctx.fillStyle = '#0f172a';
          ctx.font = 'bold 32px sans-serif';
          ctx.fillText('Machine Learning 101!', 60, 80);
          ctx.font = '20px sans-serif';
          ctx.fillText('Traditional: Data + Program -> Output', 60, 140);
          ctx.fillText('Machine Learning: Data + Output -> Program', 60, 190);
          ctx.fillText('Supervised: Classification, Regression', 60, 240);
          ctx.fillText('Unsupervised: Clustering, Association', 60, 290);
          ctx.fillText('* A ML model is only as good as the data', 60, 350);
          ctx.fillText('* Avoid Overfitting', 60, 390);
          return new Promise<Blob>((resolve) => canvas.toBlob((b) => resolve(b!), 'image/jpeg'));
        });

      const file = new File([sampleBlob], 'sample_lecture_whiteboard.jpg', { type: 'image/jpeg' });
      handleFile(file);
    } catch (err) {
      setErrorMsg('Could not load sample. Please upload your whiteboard image.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Hero Title */}
      <div className="text-center space-y-3 mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-xs font-semibold text-sky-400">
          <Sparkles className="h-3.5 w-3.5" />
          <span>AI Concept Knowledge & Whiteboard Lecture Explainer</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
          Turn Classroom Whiteboards into <span className="bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 bg-clip-text text-transparent">Smart Study Notes & Explanations</span>
        </h1>
        <p className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto">
          Upload any lecture whiteboard photo. BoardMind AI crops the board, reads messy handwriting and diagrams, and generates an in-depth academic concept explanation.
        </p>
      </div>

      {/* Upload Box */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 relative overflow-hidden group ${
            isDragging
              ? 'border-sky-400 bg-sky-500/10 scale-[1.01]'
              : selectedFile
              ? 'border-emerald-500/50 bg-slate-900/60'
              : 'border-slate-700 hover:border-slate-600 bg-slate-900/40 hover:bg-slate-900/70'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            accept="image/jpeg,image/png,image/webp,image/jpg"
            className="hidden"
          />

          {previewUrl ? (
            <div className="space-y-4">
              <div className="relative max-h-72 mx-auto inline-block rounded-xl overflow-hidden border border-slate-700 shadow-2xl">
                <img
                  src={previewUrl}
                  alt="Whiteboard preview"
                  className="max-h-64 object-contain mx-auto"
                />
                <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="text-xs font-semibold bg-slate-900/90 text-white px-3 py-1.5 rounded-lg border border-slate-700">
                    Click or drop another image to replace
                  </span>
                </div>
              </div>
              <div className="text-xs text-slate-300 font-medium">
                {selectedFile?.name} ({(selectedFile!.size / (1024 * 1024)).toFixed(2)} MB)
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 text-sky-400 flex items-center justify-center mx-auto shadow-inner group-hover:scale-110 transition-transform">
                <Upload className="h-8 w-8" />
              </div>
              <div className="space-y-1">
                <p className="text-base font-semibold text-white">
                  Drop your classroom whiteboard photo here, or <span className="text-sky-400 underline decoration-sky-400/40">browse</span>
                </p>
                <p className="text-xs text-slate-400">
                  Supports JPG, PNG, and WEBP photos up to 25MB
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Error message */}
        {errorMsg && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center gap-2 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Optional Topic Hint */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
          <div className="sm:col-span-2">
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Lecture Subject / Topic Hint (Optional)
            </label>
            <input
              type="text"
              value={topicHint}
              onChange={(e) => setTopicHint(e.target.value)}
              placeholder="e.g. Machine Learning, Data Structures, Calculus"
              className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition"
            />
          </div>

          <div className="sm:col-span-1 flex flex-col justify-end pt-5 sm:pt-0">
            <button
              type="button"
              onClick={handleLoadSample}
              className="w-full px-3 py-2.5 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-800 text-xs font-semibold text-slate-300 hover:text-white transition flex items-center justify-center gap-1.5"
            >
              <FileQuestion className="h-4 w-4 text-sky-400" />
              <span>Load Sample Demo</span>
            </button>
          </div>
        </div>

        {/* Process Button */}
        <div>
          <button
            type="submit"
            disabled={!selectedFile || isProcessing}
            className={`w-full py-3.5 px-6 rounded-2xl font-bold text-sm tracking-wide transition-all shadow-xl flex items-center justify-center gap-2 ${
              !selectedFile || isProcessing
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                : 'bg-gradient-to-r from-sky-500 to-cyan-400 hover:from-sky-400 hover:to-cyan-300 text-slate-950 shadow-sky-500/25 hover:shadow-sky-500/35 hover:scale-[1.005]'
            }`}
          >
            <span>{isProcessing ? 'Analyzing Whiteboard with AI...' : 'Detect Whiteboard & Generate AI Concept Notes'}</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  );
};

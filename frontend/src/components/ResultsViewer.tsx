import React, { useState } from 'react';
import html2pdf from 'html2pdf.js';
import {
  Copy,
  Check,
  Download,
  FileDown,
  RotateCcw,
  Sparkles,
  Layers,
  FileText,
  Edit3,
  Clock,
  ShieldCheck,
  Send
} from 'lucide-react';
import { ProcessResponse, StructuredNotes } from '../types';
import { api } from '../services/api';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ResultsViewerProps {
  result: ProcessResponse;
  onReset: () => void;
}

export const ResultsViewer: React.FC<ResultsViewerProps> = ({ result, onReset }) => {
  const [imageTab, setImageTab] = useState<'annotated' | 'cropped'>('cropped');
  const [notesTab, setNotesTab] = useState<'formatted' | 'raw' | 'edit'>('formatted');
  const [copied, setCopied] = useState<boolean>(false);
  const [isUpdatingNotes, setIsUpdatingNotes] = useState<boolean>(false);
  const [isExportingPdf, setIsExportingPdf] = useState<boolean>(false);

  // Editable state
  const [currentNotes, setCurrentNotes] = useState<StructuredNotes>(result.structured_notes);
  const [editableText, setEditableText] = useState<string>(
    result.text_extraction.raw_text ||
    result.structured_notes.raw_preserved_text ||
    (result.text_extraction.lines.length > 0
      ? result.text_extraction.lines.map(l => l.text).join('\n')
      : '')
  );

  const detection = result.detection;
  const primaryBox = detection.primary_detection;
  const strokeDensity = result.text_extraction.stroke_density;

  const handleCopy = () => {
    navigator.clipboard.writeText(currentNotes.markdown_content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = (type: 'md' | 'txt') => {
    const filename = `${currentNotes.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_notes.${type}`;
    const content = type === 'md' ? currentNotes.markdown_content : currentNotes.raw_preserved_text;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPdf = async () => {
    const element = document.getElementById('boardmind-notes-printable-area');
    if (!element) return;

    setIsExportingPdf(true);
    try {
      const cleanTitle = currentNotes.title.toLowerCase().replace(/[^a-z0-9]/g, '_');
      const filename = `${cleanTitle}_study_notes.pdf`;

      const opt = {
        margin: [12, 12, 12, 12],
        filename: filename,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: {
          scale: 2,
          useCORS: true,
          logging: false,
          backgroundColor: '#090d16'
        },
        jsPDF: {
          unit: 'mm',
          format: 'a4',
          orientation: 'portrait'
        },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
      };

      // Handle default export variation across bundlers
      const pdfWorker = typeof (html2pdf as any).default === 'function'
        ? (html2pdf as any).default
        : html2pdf;

      await pdfWorker().set(opt).from(element).save();
    } catch (err) {
      console.error('Failed to generate PDF:', err);
      alert('Could not export PDF. Please check your browser print permissions.');
    } finally {
      setIsExportingPdf(false);
    }
  };

  const handleReformatNotes = async () => {
    if (!editableText.trim()) return;
    setIsUpdatingNotes(true);
    try {
      const resp = await api.generateNotes(editableText, currentNotes.title);
      setCurrentNotes(resp.notes);
      setNotesTab('formatted');
    } catch (err) {
      alert('Failed to re-structure notes. Please check the backend connection.');
    } finally {
      setIsUpdatingNotes(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-900/90 border border-slate-800 p-4 rounded-2xl">
        <div className="flex items-center gap-3">
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-xs font-semibold text-slate-300 transition"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Process New Image</span>
          </button>
          <div className="h-4 w-px bg-slate-800" />
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Clock className="h-3.5 w-3.5 text-sky-400" />
            <span>Processed in <strong className="text-white font-mono">{result.processing_time_ms.toFixed(0)} ms</strong> on CPU</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Download PDF Button */}
          <button
            onClick={handleDownloadPdf}
            disabled={isExportingPdf}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 hover:from-emerald-400 hover:to-teal-300 text-slate-950 text-xs font-bold transition shadow-sm disabled:opacity-50"
            title="Download formatted lecture notes with mathematical formulas as a PDF document"
          >
            <FileDown className="h-3.5 w-3.5" />
            <span>{isExportingPdf ? 'Exporting PDF...' : 'Download PDF'}</span>
          </button>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 transition"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-sky-400" />
                <span>Copy Markdown</span>
              </>
            )}
          </button>

          <button
            onClick={() => handleDownload('md')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold transition shadow-sm"
          >
            <Download className="h-3.5 w-3.5" />
            <span>.MD</span>
          </button>

          <button
            onClick={() => handleDownload('txt')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 transition"
          >
            <FileText className="h-3.5 w-3.5 text-slate-400" />
            <span>.TXT</span>
          </button>
        </div>
      </div>

      {/* Two-Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT PANEL: Vision & Detection Inspection */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col">
            {/* View Switcher Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-800/40">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-sky-400" />
                <span>Computer Vision View</span>
              </span>

              <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800">
                <button
                  onClick={() => setImageTab('cropped')}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md transition ${
                    imageTab === 'cropped'
                      ? 'bg-sky-500 text-slate-950 shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Cropped Board
                </button>
                <button
                  onClick={() => setImageTab('annotated')}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md transition ${
                    imageTab === 'annotated'
                      ? 'bg-sky-500 text-slate-950 shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  YOLO Box
                </button>
              </div>
            </div>

            {/* Image Preview Container */}
            <div className="bg-slate-950 p-3 min-h-[320px] flex items-center justify-center relative">
              {imageTab === 'cropped' && result.cropped_whiteboard ? (
                <img
                  src={result.cropped_whiteboard}
                  alt="Enhanced Cropped Whiteboard"
                  className="max-h-96 w-auto object-contain rounded-lg border border-slate-800 shadow"
                />
              ) : detection.annotated_image ? (
                <img
                  src={detection.annotated_image}
                  alt="YOLO Bounding Box Detection"
                  className="max-h-96 w-auto object-contain rounded-lg border border-slate-800 shadow"
                />
              ) : (
                <div className="text-xs text-slate-500">Image not available</div>
              )}
            </div>

            {/* Metrics Footer */}
            <div className="p-4 bg-slate-900 border-t border-slate-800 space-y-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-500 block text-[11px]">YOLO Confidence</span>
                  <span className="font-semibold text-emerald-400">
                    {primaryBox ? `${(primaryBox.confidence * 100).toFixed(1)}%` : 'None'}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-500 block text-[11px]">Ink Stroke Density</span>
                  <span className="font-semibold text-cyan-400">
                    {strokeDensity !== undefined ? `${strokeDensity}% ink` : 'Measured'}
                  </span>
                </div>
              </div>

              {primaryBox && (
                <div className="text-[11px] font-mono text-slate-400 bg-slate-950/80 p-2 rounded-lg border border-slate-800">
                  <span className="text-slate-500">Bounding Box: </span>
                  [{primaryBox.xmin}, {primaryBox.ymin}, {primaryBox.xmax}, {primaryBox.ymax}] ({detection.image_width}x{detection.image_height})
                </div>
              )}
            </div>
          </div>

          {/* HTR Transparency Banner */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
            <div className="flex items-center gap-2 text-sky-400 font-semibold">
              <ShieldCheck className="h-4 w-4" />
              <span>Modular Handwriting Engine</span>
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              {result.text_extraction.message}
            </p>
          </div>
        </div>

        {/* RIGHT PANEL: Smart Study Notes Canvas */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col min-h-[580px]">
            {/* Notes Tabs */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-800/40">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setNotesTab('formatted')}
                  className={`text-xs font-bold pb-1 transition flex items-center gap-1.5 ${
                    notesTab === 'formatted'
                      ? 'text-sky-400 border-b-2 border-sky-400'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Formatted Smart Notes</span>
                </button>

                <button
                  onClick={() => setNotesTab('raw')}
                  className={`text-xs font-bold pb-1 transition flex items-center gap-1.5 ${
                    notesTab === 'raw'
                      ? 'text-sky-400 border-b-2 border-sky-400'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <FileText className="h-3.5 w-3.5" />
                  <span>Raw Text</span>
                </button>

                <button
                  onClick={() => setNotesTab('edit')}
                  className={`text-xs font-bold pb-1 transition flex items-center gap-1.5 ${
                    notesTab === 'edit'
                      ? 'text-sky-400 border-b-2 border-sky-400'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Edit3 className="h-3.5 w-3.5" />
                  <span>Editor & Refine</span>
                </button>
              </div>

              <span className="text-[11px] text-slate-500 hidden sm:inline">
                Live Markdown Engine
              </span>
            </div>

            {/* TAB CONTENT */}
            <div className="p-6 flex-1 overflow-y-auto">
              {notesTab === 'formatted' && (
                <div id="boardmind-notes-printable-area" className="space-y-6 p-2 rounded-xl">
                  {/* Title & Engine Badge */}
                  <div className="border-b border-slate-800 pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-wider text-sky-400">Subject Topic</span>
                      <h2 className="text-2xl font-bold text-white mt-1">{currentNotes.title}</h2>
                    </div>
                    {currentNotes.ai_engine_used && (
                      <span className="self-start sm:self-center px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold flex items-center gap-1.5 shadow-sm">
                        <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
                        <span>{currentNotes.ai_engine_used}</span>
                      </span>
                    )}
                  </div>

                  {/* Rendered Markdown & Real Mathematical KaTeX Formulas */}
                  {currentNotes.markdown_content ? (
                    <MarkdownRenderer content={currentNotes.markdown_content} />
                  ) : (
                    /* Fallback when no structured elements */
                    <div className="py-8 text-center text-slate-500 space-y-3">
                      <FileText className="h-8 w-8 mx-auto text-slate-600" />
                      <p className="text-xs">No formatted text recognized yet.</p>
                      <p className="text-[11px] text-slate-400 max-w-sm mx-auto">
                        Click on the <strong>"Editor & Refine"</strong> tab above to type or paste the teacher's scribbled points, and BoardMind AI will format them instantly!
                      </p>
                    </div>
                  )}
                </div>
              )}

              {notesTab === 'raw' && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-400">
                    Exact preserved text extracted from whiteboard lines:
                  </p>
                  <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
                    {currentNotes.raw_preserved_text || editableText}
                  </pre>
                </div>
              )}

              {notesTab === 'edit' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Edit Teacher Scribbles & Notes
                    </label>
                    <p className="text-[11px] text-slate-400 mb-2">
                      Adjust any ambiguous handwritten abbreviations, symbols, or points below. Then click <strong>"Re-structure Notes"</strong> to update the study notes.
                    </p>
                    <textarea
                      rows={12}
                      value={editableText}
                      onChange={(e) => setEditableText(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs font-mono text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 leading-relaxed"
                      placeholder="Type or paste whiteboard notes here..."
                    />
                  </div>

                  <button
                    onClick={handleReformatNotes}
                    disabled={isUpdatingNotes || !editableText.trim()}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-cyan-400 hover:from-sky-400 hover:to-cyan-300 text-slate-950 font-bold text-xs shadow-md transition disabled:opacity-50"
                  >
                    <Send className="h-3.5 w-3.5" />
                    <span>{isUpdatingNotes ? 'Structuring Notes...' : 'Re-structure Notes with AI'}</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

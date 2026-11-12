import React, { useMemo } from 'react';
import { marked } from 'marked';
import katex from 'katex';

interface MarkdownRendererProps {
  content: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  const renderedHtml = useMemo(() => {
    if (!content) return '';

    // Step 1: Replace block math ($$ ... $$) with KaTeX HTML
    let text = content.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
      try {
        const rendered = katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        });
        return `<div class="katex-math-block my-5 p-4 rounded-2xl bg-slate-950 border border-sky-500/30 text-center shadow-lg shadow-sky-950/40 overflow-x-auto text-sky-200">${rendered}</div>`;
      } catch (err) {
        return `<div class="p-3 my-2 bg-slate-900 rounded font-mono text-xs text-sky-400">$$${math}$$</div>`;
      }
    });

    // Step 2: Replace inline math ($...$) with KaTeX HTML (avoiding currency amounts like $100)
    text = text.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (_, math) => {
      // If it looks like a currency amount, keep as is
      if (/^\d+(\.\d{2})?$/.test(math.trim())) {
        return `$${math}`;
      }
      try {
        const rendered = katex.renderToString(math.trim(), {
          displayMode: false,
          throwOnError: false,
        });
        return `<span class="katex-math-inline px-1.5 py-0.5 rounded-md bg-slate-950/90 border border-slate-800/80 font-mono text-cyan-300 inline-block align-middle">${rendered}</span>`;
      } catch (err) {
        return `$${math}$`;
      }
    });

    // Step 3: Parse GitHub Flavored Markdown
    marked.setOptions({
      gfm: true,
      breaks: true,
    });

    return marked.parse(text) as string;
  }, [content]);

  return (
    <div
      className="boardmind-markdown-viewer"
      dangerouslySetInnerHTML={{ __html: renderedHtml }}
    />
  );
};


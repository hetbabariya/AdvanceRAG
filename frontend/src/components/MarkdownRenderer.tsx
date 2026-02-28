import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Citation } from '../api';
import './MarkdownRenderer.css';

const { useState, useEffect, useRef } = React;

interface MarkdownRendererProps {
    content: string;
    citations?: Citation[];
}

/**
 * Pre-process the answer text:
 * 1. Strip any leaked "Citations:" section the LLM may have included in the answer.
 * 2. Replace raw chunk IDs like [640a89855672abdf_chunk_400] with [N]
 *    based on the citations list.
 */
function preprocessContent(content: string, citations: Citation[]): string {
    let processed = content;

    // 1. Normalize Unicode fullwidth/CJK brackets 【N】→ [N]
    //    The LLM sometimes uses 【 】 instead of [ ]
    processed = processed.replace(/【(\d+)】/g, '[$1]');

    // 2. Strip leaked Citations section — catches all variants:
    //    "Citations:", "**Citations**", "---\n**Citations**", "Citations\n1.", etc.
    const citationsSectionPattern = /(\n[-*_]{3,}\n)?\n?\*?\*?Citations\*?\*?[:\n]/i;
    const citationsIdx = processed.search(citationsSectionPattern);
    if (citationsIdx !== -1) {
        processed = processed.slice(0, citationsIdx).trim();
    }

    if (!citations || citations.length === 0) return processed;

    // 3. Build a map from chunk_id -> citation number
    const chunkIdToNum: Record<string, number> = {};
    citations.forEach((c, idx) => {
        if (c.chunk_id) {
            chunkIdToNum[c.chunk_id] = c.number ?? (idx + 1);
        }
    });

    // 4. Replace raw [chunk_id] patterns with [N]
    processed = processed.replace(/\[([a-zA-Z0-9_]+)\]/g, (match, id) => {
        if (/^\d+$/.test(id)) return match;
        if (chunkIdToNum[id] !== undefined) return `[${chunkIdToNum[id]}]`;
        const fullKey = Object.keys(chunkIdToNum).find(k => k.includes(id) || id.includes(k));
        if (fullKey) return `[${chunkIdToNum[fullKey]}]`;
        return match;
    });

    return processed;
}

// ─── Citation Popup (click-based, stays open) ────────────────────────────────
interface CitationPopupProps {
    citation: Citation;
    num: number;
    onClose: () => void;
    anchorRef: React.RefObject<HTMLSpanElement>;
}

const CitationPopup: React.FC<CitationPopupProps> = ({ citation, num, onClose, anchorRef }) => {
    const popupRef = useRef<HTMLDivElement>(null);

    // Position popup above the badge
    const [style, setStyle] = useState<React.CSSProperties>({ visibility: 'hidden' });

    useEffect(() => {
        if (!anchorRef.current || !popupRef.current) return;
        const anchor = anchorRef.current.getBoundingClientRect();
        const popup = popupRef.current.getBoundingClientRect();
        const viewportWidth = window.innerWidth;

        let left = anchor.left + anchor.width / 2 - popup.width / 2;
        // Clamp to viewport
        if (left < 8) left = 8;
        if (left + popup.width > viewportWidth - 8) left = viewportWidth - popup.width - 8;

        const top = anchor.top - popup.height - 12;
        setStyle({ position: 'fixed', left, top, visibility: 'visible' });
    }, [anchorRef]);

    // Close on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (
                popupRef.current && !popupRef.current.contains(e.target as Node) &&
                anchorRef.current && !anchorRef.current.contains(e.target as Node)
            ) {
                onClose();
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [onClose, anchorRef]);

    // Shorten source path to just filename
    const shortSource = citation.source
        ? citation.source.split(/[\\/]/).pop() ?? citation.source
        : '';

    return (
        <div ref={popupRef} className="citation-popup" style={style}>
            {/* Arrow */}
            <div className="citation-popup-arrow" />

            {/* Header */}
            <div className="citation-popup-header">
                <div className="citation-popup-icon">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM8 13h8v1.5H8V13zm0 3h8v1.5H8V16zm0-6h3v1.5H8V10z" />
                    </svg>
                </div>
                <div className="citation-popup-title">
                    <span className="citation-popup-filename">{shortSource}</span>
                    {citation.page_number && (
                        <span className="citation-popup-page">Pg. {citation.page_number}</span>
                    )}
                </div>
                <button className="citation-popup-close" onClick={onClose}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            {/* Body */}
            <div className="citation-popup-body">
                {citation.text || 'No content available for this chunk.'}
            </div>

            {/* Footer */}
            <div className="citation-popup-footer">
                <span className="citation-popup-chunk">{citation.chunk_id}</span>
                <span className="citation-popup-num">Source [{num}]</span>
            </div>
        </div>
    );
};

// ─── Citation Badge ───────────────────────────────────────────────────────────
interface CitationBadgeProps {
    num: number;
    citation?: Citation;
}

const CitationBadge: React.FC<CitationBadgeProps> = ({ num, citation }) => {
    const [open, setOpen] = useState(false);
    const badgeRef = useRef<HTMLSpanElement>(null);

    if (!citation) {
        return <span className="citation-ref">[{num}]</span>;
    }

    return (
        <>
            <span
                ref={badgeRef}
                className={`citation-ref ${open ? 'active' : ''}`}
                onClick={(e) => { e.stopPropagation(); setOpen(o => !o); }}
            >
                [{num}]
            </span>
            {open && (
                <CitationPopup
                    citation={citation}
                    num={num}
                    onClose={() => setOpen(false)}
                    anchorRef={badgeRef}
                />
            )}
        </>
    );
};

// ─── Main Renderer ────────────────────────────────────────────────────────────
const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
    content = '',
    citations = [],
}) => {
    const processedContent = preprocessContent(content, citations);

    const renderCitations = (text: any): any => {
        if (typeof text !== 'string') return text;

        const parts = text.split(/(\[\d+\])/g);
        return parts.map((part, i) => {
            const match = part.match(/\[(\d+)\]/);
            if (match) {
                const num = parseInt(match[1]);
                const citation = citations.find(c => c.number === num) ?? citations[num - 1];
                return <CitationBadge key={i} num={num} citation={citation} />;
            }
            return part;
        });
    };

    const processChildren = (children: any): any =>
        React.Children.map(children, (child: any) =>
            typeof child === 'string' ? renderCitations(child) : child
        );

    return (
        <div className="prose prose-slate max-w-none">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    code({ inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        return !inline && match ? (
                            <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>
                                {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                        ) : (
                            <code className={className} {...props}>{children}</code>
                        );
                    },
                    p: ({ children }) => <p>{processChildren(children)}</p>,
                    li: ({ children }) => <li>{processChildren(children)}</li>,
                    strong: ({ children }) => <strong>{processChildren(children)}</strong>,
                    em: ({ children }) => <em>{processChildren(children)}</em>,
                    td: ({ children }) => <td>{processChildren(children)}</td>,
                    th: ({ children }) => <th>{processChildren(children)}</th>,
                }}
            >
                {processedContent}
            </ReactMarkdown>
        </div>
    );
};

export default MarkdownRenderer;

import React, { useState, useRef } from 'react';
import { filesAPI, api } from '../api';
import './FileUpload.css';

interface FileUploadProps {
    onFileUploaded: (fileName?: string) => void;
}

type Tab = 'file' | 'youtube';

const FileUpload: React.FC<FileUploadProps> = ({ onFileUploaded }) => {
    const [tab, setTab] = useState<Tab>('file');
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [urlInput, setUrlInput] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    // ── File upload ──────────────────────────────────────────────────────────
    const handleFile = async (file: File) => {
        const validTypes = ['.pdf', '.docx', '.txt', '.md'];
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!validTypes.includes(fileExt)) {
            setError(`Invalid type. Supported: ${validTypes.join(', ')}`);
            return;
        }
        setError('');
        setUploading(true);
        setProgress(0);
        try {
            const result = await filesAPI.uploadFile(file, (prog) => setProgress(prog));
            onFileUploaded(result.file_name);
            setProgress(100);
            setTimeout(() => { setUploading(false); setProgress(0); }, 800);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Upload failed');
            setUploading(false);
            setProgress(0);
        }
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault(); e.stopPropagation();
        setDragActive(e.type === 'dragenter' || e.type === 'dragover');
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault(); e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    };

    // ── URL ingestion ────────────────────────────────────────────────────────
    const handleUrlSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const url = urlInput.trim();
        if (!url) return;
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            setError('URL must start with http:// or https://');
            return;
        }
        const lower = url.toLowerCase();
        if (!(lower.includes('youtube.com') || lower.includes('youtu.be'))) {
            setError('Only YouTube URLs are supported');
            return;
        }
        setError('');
        setUploading(true);
        try {
            const resp = await api.post('/ingest-youtube', { url });
            onFileUploaded(resp.data?.file_name);
            setUrlInput('');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to ingest YouTube video');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="file-upload">
            {/* Tabs */}
            <div className="upload-tabs">
                <button
                    className={`upload-tab ${tab === 'file' ? 'active' : ''}`}
                    onClick={() => { setTab('file'); setError(''); }}
                >
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="13" height="13">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    File
                </button>
                <button
                    className={`upload-tab ${tab === 'youtube' ? 'active' : ''}`}
                    onClick={() => { setTab('youtube'); setError(''); }}
                >
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="13" height="13">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                    </svg>
                    YouTube
                </button>
            </div>

            {/* File Tab */}
            {tab === 'file' && (
                <div
                    className={`upload-zone ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => !uploading && fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        className="file-input"
                        accept=".pdf,.docx,.txt,.md"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                        disabled={uploading}
                    />
                    {uploading ? (
                        <div className="upload-uploading">
                            <div className="upload-spinner" />
                            <span>{progress}%</span>
                            <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${progress}%` }} />
                            </div>
                        </div>
                    ) : (
                        <>
                            <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                            </svg>
                            <span className="upload-label">Upload File</span>
                            <span className="upload-hint">PDF, DOCX, TXT, MD</span>
                        </>
                    )}
                </div>
            )}

            {/* URL Tab */}
            {tab === 'youtube' && (
                <form className="url-form" onSubmit={handleUrlSubmit}>
                    <input
                        type="url"
                        className="url-input"
                        placeholder="https://www.youtube.com/watch?v=..."
                        value={urlInput}
                        onChange={(e) => setUrlInput(e.target.value)}
                        disabled={uploading}
                    />
                    <button type="submit" className="url-submit" disabled={uploading || !urlInput.trim()}>
                        {uploading ? <div className="upload-spinner small" /> : 'Ingest'}
                    </button>
                </form>
            )}

            {error && <div className="upload-error">{error}</div>}
        </div>
    );
};

export default FileUpload;

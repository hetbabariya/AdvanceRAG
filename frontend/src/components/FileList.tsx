import React, { useState } from 'react';
import { filesAPI, FileItem } from '../api';
import './FileList.css';

interface FileListProps {
    files: FileItem[];
    selectedFile: string | null;
    onSelectFile: (fileName: string) => void;
    onFileDeleted: (fileName: string) => void;
}

const FileList: React.FC<FileListProps> = ({
    files,
    selectedFile,
    onSelectFile,
    onFileDeleted,
}) => {
    const [deleting, setDeleting] = useState<string | null>(null);

    const handleDelete = async (fileName: string, e: React.MouseEvent) => {
        e.stopPropagation();

        if (!confirm(`Are you sure you want to delete "${fileName}"?`)) {
            return;
        }

        setDeleting(fileName);
        try {
            await filesAPI.deleteFile(fileName);
            onFileDeleted(fileName);
        } catch (error) {
            console.error('Failed to delete file:', error);
            alert('Failed to delete file');
        } finally {
            setDeleting(null);
        }
    };


    if (files.length === 0) {
        return (
            <div className="file-list-empty">
                <p className="text-sm text-gray-500">No documents yet</p>
            </div>
        );
    }

    return (
        <div className="file-list">
            {files.map((file) => (
                <div
                    key={file.file_name}
                    className={`file-item ${selectedFile === file.file_name ? 'selected' : ''}`}
                    onClick={() => onSelectFile(file.file_name)}
                    title={file.file_name}
                >
                    <div className="file-icon">
                        {file.file_name.endsWith('.url') ? (
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                            </svg>
                        ) : (
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        )}
                    </div>

                    <span className="file-name">{file.file_name}</span>

                    <button
                        className="file-delete"
                        onClick={(e) => handleDelete(file.file_name, e)}
                        disabled={deleting === file.file_name}
                        title="Delete"
                    >
                        {deleting === file.file_name ? (
                            <div className="spinner" style={{ width: '12px', height: '12px' }} />
                        ) : (
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                        )}
                    </button>
                </div>
            ))}
        </div>
    );
};

export default FileList;

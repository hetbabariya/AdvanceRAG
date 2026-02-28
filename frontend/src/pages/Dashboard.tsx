import * as React from 'react';
const { useState, useEffect } = React;
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { filesAPI, FileItem } from '../api';
import FileUpload from '../components/FileUpload';
import FileList from '../components/FileList';
import ChatInterface from '../components/ChatInterface';
import './Dashboard.css';

const Dashboard: React.FC = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [files, setFiles] = useState<FileItem[]>([]);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadFiles();
    }, []);

    const loadFiles = async () => {
        try {
            const fileList = await filesAPI.getFiles();
            setFiles(fileList);
            if (fileList.length > 0 && !selectedFile) {
                setSelectedFile(fileList[0].file_name);
            }
        } catch (error) {
            console.error('Failed to load files:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    const handleFileUploaded = (fileName?: string) => {
        loadFiles();
        // If a specific file was returned (e.g., re-uploaded cached file), select it
        if (fileName) {
            setSelectedFile(fileName);
        }
    };

    const handleFileDeleted = (fileName: string) => {
        setFiles(files.filter(f => f.file_name !== fileName));
        if (selectedFile === fileName) {
            setSelectedFile(files.length > 1 ? files[0].file_name : null);
        }
    };

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="app-sidebar">
                <div className="sidebar-header">
                    <div className="logo-container">
                        <h2>AllinOneRAG</h2>
                        <span className="badge-small">v2.0</span>
                    </div>
                    <button onClick={handleLogout} className="btn-icon" title="Logout">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                    </button>
                </div>

                <div className="sidebar-content">
                    <div className="sidebar-section">
                        <FileUpload onFileUploaded={handleFileUploaded} />
                    </div>

                    <div className="sidebar-section list-section">
                        <div className="section-title">Your Documents</div>
                        {loading ? (
                            <div className="flex justify-center p-md">
                                <div className="spinner-small"></div>
                            </div>
                        ) : (
                            <FileList
                                files={files}
                                selectedFile={selectedFile}
                                onSelectFile={setSelectedFile}
                                onFileDeleted={handleFileDeleted}
                            />
                        )}
                    </div>
                </div>

                <div className="sidebar-footer">
                    <div className="user-profile">
                        <div className="avatar-placeholder">{user?.username?.charAt(0).toUpperCase()}</div>
                        <span className="username">{user?.username}</span>
                    </div>
                </div>
            </aside>

            {/* Main Content (Chat) */}
            <main className="app-main">
                {selectedFile ? (
                    <ChatInterface fileName={selectedFile} />
                ) : (
                    <div className="empty-state-full">
                        <div className="empty-content">
                            <div className="app-logo-large">
                                <h1>AllinOneRAG</h1>
                            </div>
                            <p>Select a document from the sidebar to start chatting.</p>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default Dashboard;

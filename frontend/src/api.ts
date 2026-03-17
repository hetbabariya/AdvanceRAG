import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export const api = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Interceptor to add Authorization header from localStorage if cookie fails
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('session_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Types
export interface User {
    id: number;
    username: string;
    email: string;
    created_at: string;
}

export interface FileItem {
    file_name: string;
    chunks_upserted: number;
    ingested_at: string;
    file_size?: number;
    file_hash?: string;
}

export interface ChatMessage {
    question: string;
    answer: string;
    citations: Citation[];
    cached?: boolean;
}

export interface Citation {
    source: string;
    chunk_id: string;
    page_number?: string | number;
    number?: number;
    text?: string;
}

export interface ChatResponse {
    answer: string;
    citations: Citation[];
    used_context_chunks: number;
    cached?: boolean;
}

// Auth API
export const authAPI = {
    register: async (username: string, email: string, password: string) => {
        const response = await api.post('/register', { username, email, password });
        return response.data;
    },

    login: async (username: string, password: string) => {
        const response = await api.post('/login', { username, password });
        if (response.data.session_token) {
            localStorage.setItem('session_token', response.data.session_token);
        }
        return response.data;
    },

    logout: async () => {
        const response = await api.post('/logout');
        localStorage.removeItem('session_token');
        return response.data;
    },

    getMe: async (): Promise<User> => {
        const response = await api.get('/me');
        return response.data;
    },
};

// Files API
export const filesAPI = {
    getFiles: async (): Promise<FileItem[]> => {
        const response = await api.get('/files');
        return response.data.files;
    },

    uploadFile: async (file: File, onProgress?: (progress: number) => void): Promise<{ file_name: string; cached: boolean; chunks_upserted: number }> => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/ingest', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
                if (progressEvent.total && onProgress) {
                    const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    onProgress(progress);
                }
            },
        });

        return response.data;
    },

    deleteFile: async (fileName: string) => {
        const response = await api.delete(`/files/${encodeURIComponent(fileName)}`);
        return response.data;
    },
};

// Chat API
export const chatApi = {
    sendMessage: async (
        message: string,
        fileName: string,
        useReranker: boolean = true
    ): Promise<ChatResponse> => {
        const response = await api.post('/chat', {
            message,
            file_name: fileName,
            use_reranker: useReranker,
        });
        return response.data;
    },

    sendAgentMessage: async (
        message: string,
        fileName: string,
        useReranker: boolean = true
    ): Promise<ChatResponse> => {
        const response = await api.post('/chat/agent', {
            message,
            file_name: fileName,
            use_reranker: useReranker,
        });
        return response.data;
    },

    getChatHistory: async (fileName?: string, limit: number = 50) => {
        const params = new URLSearchParams();
        if (fileName) params.append('file_name', fileName);
        params.append('limit', limit.toString());

        const response = await api.get(`/chat/history?${params.toString()}`);
        return response.data;
    },
};

// Health API
export const healthAPI = {
    check: async () => {
        const response = await api.get('/health');
        return response.data;
    },

    getCacheStats: async () => {
        const response = await api.get('/cache/stats');
        return response.data;
    },
};

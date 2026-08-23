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

// Ingestion progress
export type IngestJobStatus =
    | 'queued'
    | 'parsing'
    | 'embedding'
    | 'upserting'
    | 'completed'
    | 'failed';

export interface UploadResult {
    file_name: string;
    cached: boolean;
    chunks_upserted: number;
}

export interface IngestStatus {
    job_id: string;
    file_name: string;
    status: IngestJobStatus;
    chunks_embedded: number;
    total_chunks: number;
    error?: string | null;
    result?: (UploadResult & { file_hash: string }) | null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Percent range of the overall bar each backend pipeline stage occupies.
const STAGE_RANGE: Record<IngestJobStatus, [number, number]> = {
    queued: [13, 18],
    parsing: [18, 35],
    embedding: [35, 92],
    upserting: [92, 99],
    completed: [100, 100],
    failed: [0, 0],
};

const STAGE_LABEL: Record<IngestJobStatus, string> = {
    queued: 'Queued',
    parsing: 'Parsing document',
    embedding: 'Embedding chunks',
    upserting: 'Indexing',
    completed: 'Done',
    failed: 'Failed',
};

async function pollIngestStatus(
    jobId: string,
    emit: (pct: number, stage: string) => void,
): Promise<UploadResult> {
    let consecutiveErrors = 0;
    for (;;) {
        let data: IngestStatus;
        try {
            const resp = await api.get<IngestStatus>(`/ingest/status/${jobId}`);
            data = resp.data;
            consecutiveErrors = 0;
        } catch {
            consecutiveErrors += 1;
            if (consecutiveErrors >= 5) {
                throw new Error('Lost connection while processing the file');
            }
            await sleep(1200);
            continue;
        }

        if (data.status === 'failed') {
            throw new Error(data.error || 'Processing failed on the server');
        }

        if (data.status === 'completed') {
            emit(100, STAGE_LABEL.completed);
            return {
                file_name: data.result?.file_name ?? data.file_name,
                cached: data.result?.cached ?? false,
                chunks_upserted: data.result?.chunks_upserted ?? 0,
            };
        }

        const [lo, hi] = STAGE_RANGE[data.status] ?? [0, 0];
        let pct = lo + (hi - lo) / 2;
        let label = STAGE_LABEL[data.status];
        if (data.status === 'embedding' && data.total_chunks > 0) {
            const done = Math.min(data.chunks_embedded, data.total_chunks);
            pct = lo + ((hi - lo) * done) / data.total_chunks;
            label = `Embedding ${done}/${data.total_chunks}`;
        }
        emit(Math.round(pct), label);
        await sleep(800);
    }
}

// Files API
export const filesAPI = {
    getFiles: async (): Promise<FileItem[]> => {
        const response = await api.get('/files');
        return response.data.files;
    },

    uploadFile: async (
        file: File,
        onProgress?: (progress: number, stage: string) => void,
    ): Promise<UploadResult> => {
        const formData = new FormData();
        formData.append('file', file);

        // Monotonic reporter: never lets the bar jump backwards.
        let current = 0;
        const emit = (pct: number, stage: string) => {
            if (pct <= current && pct < 100) return;
            current = Math.min(pct, 100);
            onProgress?.(current, stage);
        };

        // Byte transfer is only the first slice; keep easing forward while the
        // server validates/stores the raw file so the bar never freezes at ~12%.
        const creepTimer = window.setInterval(() => {
            if (current < 12) return;
            emit(current + 1, 'Processing');
        }, 1500);

        try {
            const response = await api.post('/ingest', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    if (progressEvent.total) {
                        emit(Math.round((progressEvent.loaded * 100) / progressEvent.total * 0.12), 'Uploading');
                    }
                },
            });

            const data = response.data as Record<string, unknown>;

            // Async mode (INGEST_ASYNC=1): server returned 202 { job_id } — poll real progress.
            if (data && typeof data === 'object' && 'job_id' in data) {
                clearInterval(creepTimer);
                return await pollIngestStatus(data.job_id as string, emit);
            }

            // Sync mode: full result in this response.
            emit(100, STAGE_LABEL.completed);
            return data as unknown as UploadResult;
        } finally {
            clearInterval(creepTimer);
        }
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

# AllinOneRAG - Modern Document Intelligence Platform

A modern, full-stack RAG (Retrieval-Augmented Generation) system with session-based authentication, intelligent caching, and a beautiful user interface.

## 🚀 Features

### Backend
- **Session-Based Authentication**: Secure user authentication with HTTP-only cookies
- **PostgreSQL Database**: User management, session storage, and file metadata
- **Redis Caching**: 
  - Document embedding cache (30-day TTL)
  - Query result cache (1-hour TTL)
  - Session cache for fast lookups
- **Async Operations**: Non-blocking I/O for maximum performance
- **File Hash Detection**: Automatic duplicate document detection
- **Hybrid Search**: Pinecone vector search + BM25 for optimal retrieval
- **Reranking**: Cross-encoder reranking for improved relevance

### Frontend
- **Modern React UI**: Built with TypeScript and Vite
- **Glassmorphism Design**: Clean, modern aesthetic with smooth animations
- **Drag-and-Drop Upload**: Intuitive file upload with progress tracking
- **Real-Time Chat**: Interactive Q&A with document citations
- **Responsive Layout**: Works seamlessly on desktop and mobile
- **Protected Routes**: Automatic authentication flow

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Groq API Key
- Pinecone API Key

## 🛠️ Setup Instructions

### 1. Clone and Setup Environment

```bash
cd AllinOneRAG
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
SECRET_KEY=your_secret_key_here  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Start PostgreSQL and Redis

```bash
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
python -m backend.api.database
```

You should see:
```
🔄 Initializing database...
✅ Database tables created successfully!
✅ Database is ready!
```

### 5. Start Backend Server

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 6. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 7. Start Frontend Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 🎯 Usage

### First Time Setup

1. **Register an Account**
   - Navigate to `http://localhost:5173`
   - Click "Sign up" and create an account
   - You'll be automatically logged in

2. **Upload Documents**
   - Drag and drop PDF, DOCX, or HTML files
   - Or click the upload zone to browse files
   - Wait for processing (with progress indicator)

3. **Chat with Documents**
   - Select a document from the sidebar
   - Ask questions in the chat interface
   - View AI-generated answers with citations

### Features in Action

- **Duplicate Detection**: Upload the same file twice - the second upload will be instant (served from cache)
- **Query Caching**: Ask the same question twice - the second response will be instant with a ⚡ Cached badge
- **Session Persistence**: Close and reopen your browser - you'll stay logged in
- **File Management**: Delete files you no longer need

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: Async web framework
- **PostgreSQL**: Relational database (via asyncpg)
- **Redis**: In-memory cache
- **Pinecone**: Vector database
- **LangChain**: RAG orchestration
- **Groq**: LLM inference
- **Sentence Transformers**: Embeddings and reranking

### Frontend Stack
- **React 18**: UI library
- **TypeScript**: Type safety
- **Vite**: Build tool
- **React Router**: Client-side routing
- **Axios**: HTTP client
- **Vanilla CSS**: Custom design system

### Caching Strategy

```
┌─────────────────────────────────────────┐
│         File Upload Flow                │
├─────────────────────────────────────────┤
│ 1. Calculate SHA-256 hash               │
│ 2. Check Redis cache                    │
│ 3. If cached → Return instantly         │
│ 4. If not → Process & cache (30 days)  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Query Flow                      │
├─────────────────────────────────────────┤
│ 1. Hash query + file_name               │
│ 2. Check Redis cache                    │
│ 3. If cached → Return instantly         │
│ 4. If not → RAG pipeline & cache (1hr) │
└─────────────────────────────────────────┘
```

## 📊 API Endpoints

### Authentication
- `POST /register` - Create new user
- `POST /login` - Login and create session
- `POST /logout` - Logout and destroy session
- `GET /me` - Get current user info

### Files
- `GET /files` - List user's files
- `POST /ingest` - Upload and process file
- `DELETE /files/{filename}` - Delete file

### Chat
- `POST /chat` - Send message and get AI response
- `GET /chat/history` - Get chat history

### System
- `GET /health` - Health check
- `GET /cache/stats` - Cache statistics

## 🔒 Security Features

- **Password Hashing**: bcrypt with cost factor 12
- **HTTP-Only Cookies**: Session tokens not accessible via JavaScript
- **CORS Protection**: Configured for specific origins
- **Input Validation**: Pydantic schemas for all requests
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries

## 🚀 Performance Optimizations

- **Async/Await**: All I/O operations are non-blocking
- **Connection Pooling**: 20 PostgreSQL connections with 10 overflow
- **Parallel Queries**: Vector and BM25 search run concurrently
- **Embedding Cache**: Avoid re-embedding duplicate documents
- **Query Cache**: Instant responses for repeated questions
- **Lazy Loading**: Models loaded once and reused

## 📈 Monitoring

Check cache performance:
```bash
curl http://localhost:8000/cache/stats
```

Response:
```json
{
  "keyspace_hits": 150,
  "keyspace_misses": 50,
  "total_keys": 25,
  "hit_rate": 0.75
}
```

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps

# View logs
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Redis Connection Issues
```bash
# Check if Redis is running
docker-compose ps

# Test Redis connection
docker-compose exec redis redis-cli ping
```

### Frontend Build Issues
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 📝 License

MIT License - feel free to use this project for learning and development!

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **Pinecone**: Vector database
- **Groq**: Fast LLM inference
- **FastAPI**: Modern Python web framework
- **React**: UI library

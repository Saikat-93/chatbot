# ✦ Gemini Chat — ChatGPT-style Clone

A beautiful, full-featured AI chat interface powered by **Google Gemini**, **FastAPI**, **Mem0** for persistent memory, and **UV** for package management.

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- [UV](https://github.com/astral-sh/uv) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Google Gemini API key — [Get one here](https://aistudio.google.com/app/apikey)
- *(Optional)* Qdrant for full Mem0 support — `docker run -p 6333:6333 qdrant/qdrant`

---

### 1. Configure API Key

```bash
cd backend
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

---

### 2. Install & Run Backend (with UV)

```bash
cd backend

# Create venv and install deps
uv venv
uv sync

# OR install manually
uv pip install fastapi uvicorn google-generativeai mem0ai python-dotenv

# Run the server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### 3. Open Frontend

Just open `frontend/index.html` in your browser — or serve it:

```bash
cd frontend
python -m http.server 3000
# Visit: http://localhost:3000
```

---

## 🧠 How Memory Works

- **Mem0 + Qdrant** (if available): Persistent vector memory across sessions. The AI extracts facts from conversations and stores them semantically.
- **Fallback**: If Qdrant isn't running, an in-memory store is used (resets on server restart).
- Memory is scoped per session + user, so each conversation has its own context.
- The AI uses memories to answer questions based on history: *"What did I tell you about my job?"* → it knows!

---

## 🗂 Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat sessions** | Create, rename, delete, switch between conversations |
| 🧠 **Memory** | Mem0 stores facts from each session; AI learns over time |
| ⚡ **Streaming** | Real-time token streaming via SSE |
| 🎨 **Beautiful UI** | Dark theme, smooth animations, code formatting |
| 📱 **Responsive** | Works on mobile too |

---

## 📁 Project Structure

```
gemini-chat/
├── backend/
│   ├── main.py          # FastAPI app with Gemini + Mem0
│   ├── pyproject.toml   # UV dependencies
│   └── .env.example     # Environment template
└── frontend/
    └── index.html       # Single-file chat UI
```

---

## 🐳 Optional: Qdrant with Docker

For full persistent memory:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---------|-------------|
| `GEMINI_API_KEY` | Your Google Gemini API key (required) |

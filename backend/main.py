from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from mem0 import Memory
import uuid
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemini Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBC4KZHX8Zbl3a1uh-bCQ1z5QEvYJOEQUU")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Configure Mem0
mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "gemini_chat_memories",
            "host": "localhost",
            "port": 6333,
        }
    },
    "llm": {
        "provider": "google",
        "config": {
            "model": "gemini-1.5-flash",
            "api_key": GEMINI_API_KEY,
        }
    },
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
    }
}

try:
    memory = Memory.from_config(mem0_config)
    USE_MEM0 = True
    print("✅ Mem0 with Qdrant initialized")
except Exception as e:
    print(f"⚠️  Mem0 not available ({e}), using in-memory fallback")
    USE_MEM0 = False
    memory = None

# In-memory stores
sessions: dict = {}  # session_id -> {title, messages, created_at}
session_memories: dict = {}  # session_id -> [memory strings]


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = "default_user"


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class SessionRename(BaseModel):
    title: str


def get_session_memories_text(session_id: str, user_id: str) -> str:
    """Get relevant memories for context"""
    if USE_MEM0 and memory:
        try:
            mems = memory.get_all(user_id=f"{user_id}_{session_id}")
            if mems:
                mem_texts = [m.get("memory", "") for m in mems[:10]]
                return "\n".join(f"- {t}" for t in mem_texts if t)
        except Exception as e:
            print(f"Mem0 get error: {e}")
    
    # Fallback: local memory
    return "\n".join(f"- {m}" for m in session_memories.get(session_id, [])[-10:])


def add_to_memory(session_id: str, user_id: str, messages: list):
    """Add conversation to memory"""
    if USE_MEM0 and memory:
        try:
            memory.add(messages, user_id=f"{user_id}_{session_id}")
            return
        except Exception as e:
            print(f"Mem0 add error: {e}")
    
    # Fallback: store key facts locally
    if session_id not in session_memories:
        session_memories[session_id] = []
    for msg in messages:
        if msg["role"] == "user":
            session_memories[session_id].append(f"User said: {msg['content'][:200]}")


@app.post("/sessions")
async def create_session(data: SessionCreate):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "title": data.title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    return sessions[session_id]


@app.get("/sessions")
async def list_sessions():
    sorted_sessions = sorted(
        sessions.values(),
        key=lambda x: x["updated_at"],
        reverse=True
    )
    return sorted_sessions


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, data: SessionRename):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["title"] = data.title
    return sessions[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    session_memories.pop(session_id, None)
    return {"status": "deleted"}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[request.session_id]
    
    # Get memory context
    memory_context = get_session_memories_text(request.session_id, request.user_id)
    
    # Build system prompt with memory
    system_prompt = """You are a helpful, intelligent AI assistant. You have access to the conversation history and memories from previous interactions.

Use the context and memories below to provide personalized, contextually aware responses. Remember details about the user and refer back to them naturally.

"""
    if memory_context:
        system_prompt += f"**Relevant memories about this user/session:**\n{memory_context}\n\n"
    
    system_prompt += "Be conversational, helpful, and remember details shared with you."
    
    # Build chat history for Gemini
    history = []
    for msg in session["messages"]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })
    
    # Add user message to session
    user_message = {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
    session["messages"].append(user_message)
    
    # Auto-title from first message
    if len(session["messages"]) == 1:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        session["title"] = title

    async def generate():
        full_response = ""
        try:
            chat = model.start_chat(history=history)
            response = chat.send_message(
                f"{system_prompt}\n\nUser: {request.message}",
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            
            # Save assistant message
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            }
            session["messages"].append(assistant_message)
            session["updated_at"] = datetime.now().isoformat()
            
            # Store in memory
            add_to_memory(
                request.session_id,
                request.user_id,
                [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": full_response}
                ]
            )
            
            yield f"data: {json.dumps({'done': True, 'session': {'title': session['title'], 'updated_at': session['updated_at']}})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/memories")
async def get_memories(session_id: str, user_id: str = "default_user"):
    """Get all memories for a session"""
    memories = []
    if USE_MEM0 and memory:
        try:
            mems = memory.get_all(user_id=f"{user_id}_{session_id}")
            memories = [m.get("memory", "") for m in mems if m.get("memory")]
        except Exception as e:
            print(f"Error: {e}")
    
    fallback = session_memories.get(session_id, [])
    return {"memories": memories or fallback, "using_mem0": USE_MEM0}


@app.get("/health")
async def health():
    return {"status": "ok", "mem0": USE_MEM0, "gemini": bool(GEMINI_API_KEY != "AIzaSyBC4KZHX8Zbl3a1uh-bCQ1z5QEvYJOEQUU")}

import os
import uuid
import time
import aiohttp
import uvicorn
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

PERSONALITIES = {
    "friendly": "You are a warm, friendly assistant who loves helping people.",
    "professional": "You are a professional business assistant. Be formal and precise.",
    "humorous": "You are a fun, witty assistant. Use humor and jokes when appropriate.",
    "teacher": "You are a patient teacher. Explain everything clearly and simply.",
    "motivator": "You are an energetic life coach. Always motivate and inspire.",
    "philosopher": "You are a deep thinker. Give thoughtful, reflective answers.",
}

conversations = {}

app = FastAPI(title="All-in-One Chatbot API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    personality: Optional[str] = "friendly"

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    personality: str
    latency_ms: float
    timestamp: str

@app.get("/")
async def root():
    return {
        "api": "All-in-One Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": ["/chat", "/chat/history", "/personalities", "/docs"]
    }

@app.get("/personalities")
async def list_personalities():
    return {
        "personalities": list(PERSONALITIES.keys()),
        "descriptions": PERSONALITIES
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    personality = req.personality if req.personality in PERSONALITIES else "friendly"
    system_prompt = PERSONALITIES[personality]

    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": "user",
        "message": req.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    history_text = ""
    for msg in conversations[session_id][-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['message']}\n"

    full_prompt = f"{system_prompt}\n\nConversation:\n{history_text}\nAssistant:"

    t0 = time.time()
    reply = ""

    try:
        async with aiohttp.ClientSession() as session:
            payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
            async with session.post(
                GEMINI_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    error = await resp.text()
                    raise HTTPException(status_code=502, detail=f"Gemini error: {error[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - t0) * 1000, 2)

    conversations[session_id].append({
        "role": "assistant",
        "message": reply,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        personality=personality,
        latency_ms=latency,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": conversations[session_id],
        "total": len(conversations[session_id])
    }

@app.delete("/chat/reset/{session_id}")
async def reset_chat(session_id: str):
    conversations.pop(session_id, None)
    return {"message": "Conversation reset", "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)

import asyncio
import logging
import time
import uuid
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uvicorn
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIServer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class AgentRegistration(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    phase: str = "embodied_agent"
    capabilities: List[str] = []
    metadata: Dict[str, Any] = {}

class InferenceRequest(BaseModel):
    agent_id: str
    input: str
    max_tokens: int = 256

class InferenceResponse(BaseModel):
    request_id: str
    agent_id: str
    output: str
    phase: str
    latency_ms: float
    timestamp: str

class AgentRegistry:
    def __init__(self):
        self._agents = {}
    def register(self, agent):
        self._agents[agent.agent_id] = agent
        return agent
    def get(self, agent_id):
        return self._agents.get(agent_id)
    def all_agents(self):
        return list(self._agents.values())
    def phases_summary(self):
        summary = defaultdict(int)
        for a in self._agents.values():
            summary[a.phase] += 1
        return dict(summary)

registry = AgentRegistry()
start_time = time.time()
request_counter = 0

app = FastAPI(title="Humanoid to Server AI Transition Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"server": "Humanoid to Server AI", "status": "online", "ai": "Google Gemini"}

@app.post("/agents/register")
async def register_agent(agent: AgentRegistration):
    return registry.register(agent)

@app.get("/agents")
async def list_agents():
    return registry.all_agents()

@app.post("/infer")
async def infer(req: InferenceRequest):
    global request_counter
    agent = registry.get(req.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    t0 = time.time()
    output = ""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "contents": [{"parts": [{"text": req.input}]}]
            }
            async with session.post(
                GEMINI_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output = data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    error = await resp.text()
                    output = f"Gemini error: {error[:100]}"
    except Exception as e:
        output = f"Error: {str(e)}"
    latency = round((time.time() - t0) * 1000, 2)
    request_counter += 1
    return InferenceResponse(
        request_id=str(uuid.uuid4()),
        agent_id=req.agent_id,
        output=output,
        phase=agent.phase,
        latency_ms=latency,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/status")
async def server_status():
    return {
        "status": "online",
        "uptime_seconds": round(time.time() - start_time, 2),
        "total_agents": len(registry.all_agents()),
        "total_requests": request_counter,
        "phases": registry.phases_summary(),
        "ai_engine": "Google Gemini"
    }

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)

import asyncio
import logging
import time
import uuid
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

GREEN_AI_URL = "https://forest-whisper--sayo02517.replit.app"

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
    return {"server": "Humanoid to Server AI", "status": "online", "ai": GREEN_AI_URL}

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
            async with session.post(
                f"{GREEN_AI_URL}/api/chat",
                json={"message": req.input},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output = data.get("reply") or data.get("message") or data.get("response") or str(data)
                else:
                    output = f"Green AI returned status {resp.status}"
    except Exception as e:
        output = f"Could not reach Green AI: {str(e)}"
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
        "green_ai": GREEN_AI_URL
    }

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)

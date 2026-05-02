import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIServer")

class TransitionPhase(str):
    EMBODIED = "embodied_agent"
    EDGE = "edge_offload"
    THIN = "thin_client"
    VIRTUAL = "virtual_shell"
    SERVER = "server_ai"

class AgentRegistration(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    phase: str = "embodied_agent"
    capabilities: List[str] = []
    metadata: Dict[str, Any] = {}

class InferenceRequest(BaseModel):
    agent_id: str
    input: str
    context: Optional[Dict[str, Any]] = {}
    max_tokens: int = 256
    priority: int = 1

class InferenceResponse(BaseModel):
    request_id: str
    agent_id: str
    output: str
    phase: str
    latency_ms: float
    timestamp: str
    tokens_used: int

class CentralMemoryStore:
    def __init__(self):
        self._store = defaultdict(dict)
        self._expiry = defaultdict(dict)

    def write(self, agent_id, key, value, ttl=None):
        self._store[agent_id][key] = value
        if ttl:
            self._expiry[agent_id][key] = time.time() + ttl

    def read(self, agent_id, key):
        expiry = self._expiry[agent_id].get(key)
        if expiry and time.time() > expiry:
            del self._store[agent_id][key]
            return None
        return self._store[agent_id].get(key)

    def read_all(self, agent_id):
        return dict(self._store.get(agent_id, {}))

    def delete(self, agent_id, key):
        self._store[agent_id].pop(key, None)

class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register(self, agent):
        self._agents[agent.agent_id] = agent
        return agent

    def get(self, agent_id):
        return self._agents.get(agent_id)

    def update_phase(self, agent_id, phase):
        if agent_id not in self._agents:
            return None
        self._agents[agent_id].phase = phase
        return self._agents[agent_id]

    def all_agents(self):
        return list(self._agents.values())

    def phases_summary(self):
        summary = defaultdict(int)
        for a in self._agents.values():
            summary[a.phase] += 1
        return dict(summary)

registry = AgentRegistry()
memory = CentralMemoryStore()
start_time = time.time()
request_counter = 0

app = FastAPI(title="Humanoid to Server AI Transition Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"server": "Humanoid to Server AI", "status": "online"}

@app.post("/agents/register")
async def register_agent(agent: AgentRegistration):
    return registry.register(agent)

@app.get("/agents")
async def list_agents():
    return registry.all_agents()

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = registry.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/transition")
async def transition_agent(agent_id: str, target_phase: str):
    agent = registry.update_phase(agent_id, target_phase)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/infer")
async def infer(req: InferenceRequest):
    global request_counter
    agent = registry.get(req.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    t0 = time.time()
    await asyncio.sleep(0.05)
    output = f"[{agent.phase}] Response to: {req.input[:50]}"
    latency = round((time.time() - t0) * 1000, 2)
    request_counter += 1
    memory.write(req.agent_id, "last_request", req.input)
    return InferenceResponse(
        request_id=str(uuid.uuid4()),
        agent_id=req.agent_id,
        output=output,
        phase=agent.phase,
        latency_ms=latency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tokens_used=len(output.split())
    )

@app.get("/status")
async def server_status():
    return {
        "status": "online",
        "uptime_seconds": round(time.time() - start_time, 2),
        "total_agents": len(registry.all_agents()),
        "total_requests": request_counter,
        "phases": registry.phases_summary()
    }

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)

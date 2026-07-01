import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent.agent import MovieRecommenderAgent

app = FastAPI(title="Movie Recommender Agent API")

# Mount frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Single agent instance shared across requests
agent = MovieRecommenderAgent()

# In-memory conversation store per session
# Simple for now - keyed by session_id string
conversations: dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Get or create conversation history for this session
    if request.session_id not in conversations:
        conversations[request.session_id] = []

    history = conversations[request.session_id]

    # Get agent response
    response = await agent.chat(request.message, history)

    # Update history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response})

    # Keep history bounded - sliding window of last 20 messages
    if len(history) > 20:
        conversations[request.session_id] = history[-20:]

    return ChatResponse(response=response, session_id=request.session_id)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
    return {"status": "cleared", "session_id": session_id}
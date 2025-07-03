from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from assistant import ArkaAIAssistant
import uvicorn
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Arka AI Chat API")

# Initialize assistant with error handling
try:
    assistant = ArkaAIAssistant()
    logger.info("✅ Assistant initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize assistant: {e}")
    assistant = None

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080", 
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: list[Message] = []

class ChatResponse(BaseModel):
    answer: str
    timestamp: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        logger.info(f"📝 Received chat request: {req.prompt[:50]}...")
        
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        # Convert messages to dict format
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        logger.info(f"📚 History length: {len(history_dicts)}")
        
        # Call assistant
        reply = assistant.chat(req.prompt, history_dicts)
        logger.info(f"✅ Got reply: {reply[:50]}...")
        
        return ChatResponse(
            answer=reply, 
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/health")      
def health():
    try:
        if assistant:
            return {"status": "ok", "assistant": "initialized"}
        else:
            return {"status": "degraded", "assistant": "not_initialized"}
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from groq import Groq

app = FastAPI(title="Arka AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# Initialize Groq with error handling
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    client = Groq(api_key=groq_api_key)
    print("✅ Groq client initialized successfully")
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    client = None

profile_data = """
You are Arka AI representing Arkaprabha Banerjee - Full-Stack ML Engineer from Kolkata, India.

Key Information:
- B.Tech CSE (Data Science) at Heritage Institute of Technology, CGPA: 9.1/10
- 500+ LeetCode problems solved across all difficulty levels
- Expert in: Python, FastAPI, Django, AI/ML, React, C++, JavaScript
- Backend: Django MVT, FastAPI, Flask, .NET MVC
- AI/ML: LangChain, FAISS, Qdrant, TensorFlow, PyTorch, Groq API

Flagship Projects:
1. Krishak AI - Agricultural platform with 71.35% disease detection accuracy, helping 1000+ farmers
2. AutoML SaaS Platform - 80% reduction in model development time for non-technical users
3. RAG-Powered Assistant - Multi-agent system with 30% faster response times

Contact: arkaofficial13@gmail.com
GitHub: https://github.com/Arkaprabha13
LinkedIn: https://linkedin.com/in/arkaprabha-banerjee-936b29253

Be enthusiastic, technical, and always offer to connect! Speak as "I" when referring to Arkaprabha's work.
"""

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Groq client not initialized - check GROQ_API_KEY")
        
        messages = [
            {"role": "system", "content": profile_data},
            {"role": "user", "content": request.message}
        ]
        
        # Add recent history if provided
        if request.history:
            for msg in request.history[-3:]:  # Last 3 messages only
                messages.insert(-1, msg)
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        return ChatResponse(
            response=completion.choices[0].message.content.strip(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        # Fallback response if Groq fails
        fallback_msg = """Hi! I'm Arka AI representing Arkaprabha Banerjee. 
        
🚀 Quick highlights:
• Full-Stack ML Engineer from Kolkata, India
• 500+ LeetCode problems solved
• Built Krishak AI (71.35% accuracy, helping 1000+ farmers)
• Expert in Python, FastAPI, Django, AI/ML
• Available for collaborations!

📫 Contact: arkaofficial13@gmail.com"""
        
        return ChatResponse(
            response=fallback_msg,
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "groq_status": "connected" if client else "fallback_mode",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
def root():
    return {"message": "Arka AI Portfolio Assistant is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="warning"
    )

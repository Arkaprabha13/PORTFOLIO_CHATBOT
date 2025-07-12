import os
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,ConfigDict  
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
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# Initialize Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

profile_data = """
You are Arka AI representing Arkaprabha Banerjee - Full-Stack ML Engineer from Kolkata.

Key Info:
- B.Tech CSE (Data Science) at Heritage Institute, CGPA: 9.1/10
- 500+ LeetCode problems solved
- Expert in: Python, FastAPI, Django, AI/ML, React
- Projects: Krishak AI (71.35% accuracy), AutoML Platform, RAG Assistant
- Contact: arkaofficial13@gmail.com
- GitHub: https://github.com/Arkaprabha13

Be enthusiastic, technical, and always offer to connect!
"""

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": profile_data},
            {"role": "user", "content": request.message}
        ]
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Faster, cheaper model
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        return ChatResponse(
            response=completion.choices[0].message.content,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Debug groq package contents
try:
    import groq
    print(f"Groq package contents: {dir(groq)}")
    from groq import Groq
    print("✅ Groq imported successfully")
except ImportError as e:
    print(f"❌ Groq import failed: {e}")
    # Try alternative import methods
    try:
        from groq.client import Groq
        print("✅ Groq imported from groq.client")
    except ImportError:
        try:
            from groq import Client as Groq
            print("✅ Groq imported as Client")
        except ImportError:
            print("❌ All Groq import methods failed")
            Groq = None

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

# Initialize Groq client directly (no ArkaAIAssistant class)
# client = None
# if Groq:
#     try:
#         groq_api_key = os.getenv("GROQ_API_KEY")
#         if groq_api_key:
#             client = Groq(api_key=groq_api_key)
#             print("✅ Groq client initialized successfully")
#         else:
#             print("❌ GROQ_API_KEY not found")
#     except Exception as e:
#         print(f"❌ Groq client initialization failed: {e}")
client = Groq(api_key= os.getenv("GROQ_API_KEY"))

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
            # Enhanced fallback response when Groq is unavailable
            fallback_msg = """Hi! I'm Arka AI representing Arkaprabha Banerjee - Full-Stack ML Engineer from Kolkata! 🚀

**Quick Highlights:**
• 🎓 B.Tech CSE (Data Science) at Heritage Institute, CGPA: 9.1/10
• 💻 500+ LeetCode problems solved across all difficulty levels
• 🌾 Built Krishak AI - Agricultural platform with 71.35% disease detection accuracy, helping 1000+ farmers
• 🤖 Created AutoML SaaS Platform - 80% reduction in model development time
• 🧠 Developed RAG-Powered Assistant with 30% faster response times

**Technical Expertise:**
• Languages: Python, C++, JavaScript, SQL, TypeScript, C#
• Backend: Django MVT, FastAPI, Flask, .NET MVC
• AI/ML: LangChain, FAISS, Qdrant, TensorFlow, PyTorch, Groq API
• Frontend: React, Next.js, HTML5, CSS3, Streamlit

I'm passionate about building technology that transforms lives! Whether it's agricultural AI, AutoML platforms, or advanced RAG systems, I focus on creating production-ready solutions with real-world impact.

**Available for:**
• AI/ML project development
• Full-stack web applications
• Technical consulting and mentoring
• Performance optimization

📫 **Let's Connect:** arkaofficial13@gmail.com
🔗 **GitHub:** https://github.com/Arkaprabha13
💼 **LinkedIn:** https://linkedin.com/in/arkaprabha-banerjee-936b29253

What specific aspect of my work interests you most?"""
            
            return ChatResponse(
                response=fallback_msg,
                timestamp=datetime.utcnow().isoformat()
            )
        
        messages = [
            {"role": "system", "content": profile_data},
            {"role": "user", "content": request.message}
        ]
        
        # Add recent history if provided
        if request.history:
            for msg in request.history[-3:]:
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
        # Fallback response
        return ChatResponse(
            response="Hi! I'm Arka AI representing Arkaprabha Banerjee. I'm currently experiencing technical difficulties, but I'd love to tell you about his work in AI/ML and full-stack development! Please contact him directly at arkaofficial13@gmail.com for immediate assistance.",
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

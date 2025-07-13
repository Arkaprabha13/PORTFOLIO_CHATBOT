import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Clear all proxy environment variables
proxy_vars = [
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
    'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy',
    'FTP_PROXY', 'ftp_proxy', 'SOCKS_PROXY', 'socks_proxy',
    'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE'
]

for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]
        print(f"Cleared {var}")

# Force disable proxy detection
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

# Patch HTTP libraries BEFORE importing Groq
def patch_http_libraries():
    """Patch HTTP libraries to prevent proxy parameter passing"""
    try:
        import httpx
        original_client_init = httpx.Client.__init__
        
        def patched_client_init(self, *args, **kwargs):
            # Remove proxy-related parameters
            kwargs.pop('proxies', None)
            kwargs.pop('proxy', None)
            kwargs.pop('trust_env', None)
            # Force disable environment trust
            kwargs['trust_env'] = False
            return original_client_init(self, *args, **kwargs)
        
        httpx.Client.__init__ = patched_client_init
        print("✅ HTTPX patched successfully")
    except ImportError:
        print("⚠️ HTTPX not available for patching")
    
    try:
        import requests
        original_session_init = requests.Session.__init__
        
        def patched_session_init(self, *args, **kwargs):
            result = original_session_init(self, *args, **kwargs)
            # Clear any proxy settings
            self.proxies = {}
            self.trust_env = False
            return result
        
        requests.Session.__init__ = patched_session_init
        print("✅ Requests patched successfully")
    except ImportError:
        print("⚠️ Requests not available for patching")

# Apply patches BEFORE importing Groq
patch_http_libraries()

# Now import Groq
from groq import Groq

# Initialize Groq client
print("=== Initializing Groq Client ===")
groq_api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY exists: {bool(groq_api_key)}")
print(f"GROQ_API_KEY length: {len(groq_api_key) if groq_api_key else 0}")
print(f"PORT: {os.environ.get('PORT', 'Not set')}")

try:
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    client = Groq(api_key=groq_api_key)
    print("✅ Groq client initialized successfully")
    
    # Test the client with qwen model
    try:
        test_response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
            temperature=0.7
        )
        print("✅ Groq client test with qwen/qwen3-32b successful")
    except Exception as test_error:
        print(f"⚠️ Groq test failed: {test_error}")
        # Client is still usable even if test fails
    
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    print(f"❌ Error type: {type(e).__name__}")
    client = None

# Initialize FastAPI app
app = FastAPI(title="Arka AI Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# Profile data for the AI assistant
profile_data = """
You are Arka AI representing Arkaprabha Banerjee - Full-Stack ML Engineer from Kolkata, India.

Key Information:
- B.Tech CSE (Data Science) at Heritage Institute of Technology, CGPA: 9.1/10
- 500+ LeetCode problems solved across all difficulty levels
- Expert in: Python, FastAPI, Django, AI/ML, React, C++, JavaScript
- Backend: Django MVT, FastAPI, Flask, .NET MVC
- AI/ML: LangChain, FAISS, Qdrant, TensorFlow, PyTorch, Groq API
- Frontend: React, Next.js, HTML5, CSS3, Streamlit

Flagship Projects:
1. Krishak AI - Agricultural platform with 71.35% disease detection accuracy, helping 1000+ farmers
2. AutoML SaaS Platform - 80% reduction in model development time for non-technical users
3. RAG-Powered Assistant - Multi-agent system with 30% faster response times

Contact: arkaofficial13@gmail.com
GitHub: https://github.com/Arkaprabha13
LinkedIn: https://linkedin.com/in/arkaprabha-banerjee-936b29253

Be enthusiastic, technical, and always offer to connect! Speak as "I" when referring to Arkaprabha's work.
"""

# API Routes
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages and returns AI responses
    """
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
        
        # Prepare messages for the AI
        messages = [
            {"role": "system", "content": profile_data},
            {"role": "user", "content": request.message}
        ]
        
        # Add recent history if provided (last 3 messages)
        if request.history:
            for msg in request.history[-3:]:
                messages.insert(-1, msg)
        
        # Call Groq API with qwen model
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.95,
            stop=None
        )
        
        return ChatResponse(
            response=completion.choices[0].message.content.strip(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        # Fallback response for any errors
        return ChatResponse(
            response="Hi! I'm Arka AI representing Arkaprabha Banerjee. I'm currently experiencing technical difficulties, but I'd love to tell you about his work in AI/ML and full-stack development! Please contact him directly at arkaofficial13@gmail.com for immediate assistance.",
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/api/health")
def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "groq_status": "connected" if client else "fallback_mode",
        "model": "qwen/qwen3-32b",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
def root():
    """
    Root endpoint
    """
    return {
        "message": "Arka AI Portfolio Assistant is running!",
        "model": "qwen/qwen3-32b",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

@app.get("/api/info")
def info():
    """
    Information endpoint about the assistant
    """
    return {
        "name": "Arka AI Assistant",
        "representing": "Arkaprabha Banerjee",
        "model": "qwen/qwen3-32b",
        "specialties": [
            "Full-Stack Development",
            "Machine Learning",
            "AI/ML Solutions",
            "Agricultural Technology",
            "AutoML Platforms"
        ],
        "contact": {
            "email": "arkaofficial13@gmail.com",
            "github": "https://github.com/Arkaprabha13",
            "linkedin": "https://linkedin.com/in/arkaprabha-banerjee-936b29253"
        }
    }

# Main execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Arka AI Assistant on port {port}")
    print(f"Using model: qwen/qwen3-32b")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

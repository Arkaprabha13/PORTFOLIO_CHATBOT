import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import logging

# Lightweight imports for minimal memory footprint
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure minimal logging to save memory
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arka AI Portfolio Assistant",
    description="Lightweight chatbot for Arkaprabha Banerjee's portfolio",
    version="1.0.0"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# Lightweight Portfolio Assistant (No external API calls)
class ArkaPortfolioAssistant:
    def __init__(self):
        self.profile = {
            "name": "Arkaprabha Banerjee",
            "role": "Full-Stack ML Engineer & Data Science Enthusiast",
            "location": "Kolkata, India",
            "education": "B.Tech CSE (Data Science) at Heritage Institute of Technology",
            "cgpa": "9.1/10 (6th Semester)",
            "contact": {
                "email": "arkaofficial13@gmail.com",
                "github": "https://github.com/Arkaprabha13",
                "linkedin": "https://linkedin.com/in/arkaprabha-banerjee-936b29253",
                "leetcode": "https://leetcode.com/arkaofficial13/",
                "kaggle": "https://www.kaggle.com/arkaprabhabanerjee13"
            },
            "skills": {
                "languages": ["Python", "C++", "JavaScript", "SQL", "TypeScript", "C#"],
                "backend": ["Django", "FastAPI", "Flask", ".NET MVC"],
                "frontend": ["React", "Next.js", "HTML5", "CSS3", "Streamlit"],
                "ai_ml": ["LangChain", "FAISS", "Qdrant", "YOLOv5", "TensorFlow", "PyTorch", "scikit-learn"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Vector DBs"],
                "tools": ["Docker", "Kubernetes", "GitHub Actions", "OpenCV", "MLflow"]
            },
            "projects": {
                "krishak_ai": {
                    "name": "Krishak - Agricultural AI Platform",
                    "description": "Revolutionary agriculture platform with 71.35% disease detection accuracy",
                    "tech": ["Python", "FastAPI", "CNN", "Computer Vision", "Twilio"],
                    "impact": "Helping 1000+ farmers with real-time crop recommendations",
                    "achievements": ["71.35% accuracy", "3-second response time", "SMS integration"]
                },
                "automl_platform": {
                    "name": "AutoML SaaS Platform",
                    "description": "End-to-end automated ML platform for non-technical users",
                    "tech": ["Python", "Streamlit", "scikit-learn", "Flask"],
                    "impact": "80% reduction in model development time",
                    "achievements": ["Automated EDA", "Parallel model training", "Easy deployment"]
                },
                "rag_assistant": {
                    "name": "RAG-Powered Multi-Agent Assistant",
                    "description": "Sophisticated knowledge assistant with multi-agent architecture",
                    "tech": ["Python", "LangChain", "FAISS", "Groq API", "FastAPI"],
                    "impact": "30% faster response time with advanced vector search",
                    "achievements": ["Context-aware responses", "Memory persistence", "Agent routing"]
                }
            },
            "achievements": [
                "500+ LeetCode problems solved",
                "CGPA: 9.1/10 in Data Science",
                "Built 25+ production-ready projects",
                "Mentored 50+ students",
                "Active open-source contributor"
            ],
            "availability": "Available for new projects and collaborations"
        }
        
        # Pre-defined responses for common queries (saves API calls)
        self.quick_responses = {
            "greeting": "Hi! I'm Arka AI, representing Arkaprabha Banerjee. I'm excited to tell you about his work in AI/ML and full-stack development! What would you like to know?",
            "skills": "I have expertise in Python, C++, JavaScript, and more. My backend skills include Django, FastAPI, Flask, and .NET MVC. I'm passionate about AI/ML with experience in LangChain, TensorFlow, PyTorch, and vector databases.",
            "projects": "My flagship projects include Krishak AI (agricultural platform with 71.35% accuracy), AutoML SaaS platform (80% faster development), and RAG-powered assistants. Each project focuses on real-world impact!",
            "contact": "You can reach me at arkaofficial13@gmail.com or connect on LinkedIn, GitHub, or LeetCode. I'm always excited to discuss new opportunities!",
            "education": "I'm pursuing B.Tech CSE (Data Science) at Heritage Institute of Technology with a 9.1/10 CGPA. I believe in learning by building real-world solutions!",
            "availability": "I'm available for AI/ML projects, full-stack development, technical consulting, and mentoring. I respond within 24 hours and love collaborating on impactful solutions!"
        }

    def generate_response(self, query: str, history: List[Dict] = None) -> str:
        """Generate response using pattern matching and templates"""
        query_lower = query.lower()
        
        # Quick pattern matching for common queries
        if any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            return self.quick_responses["greeting"]
        
        if any(word in query_lower for word in ['skill', 'technology', 'tech', 'expertise', 'stack']):
            return self._format_skills_response()
        
        if any(word in query_lower for word in ['project', 'work', 'portfolio', 'built', 'created']):
            return self._format_projects_response()
        
        if any(word in query_lower for word in ['contact', 'email', 'reach', 'connect', 'linkedin', 'github']):
            return self._format_contact_response()
        
        if any(word in query_lower for word in ['education', 'study', 'college', 'university', 'cgpa']):
            return self._format_education_response()
        
        if any(word in query_lower for word in ['hire', 'available', 'collaboration', 'opportunity']):
            return self._format_availability_response()
        
        if any(word in query_lower for word in ['krishak', 'agriculture', 'farming']):
            return self._format_krishak_response()
        
        if any(word in query_lower for word in ['automl', 'machine learning', 'ml']):
            return self._format_automl_response()
        
        if any(word in query_lower for word in ['rag', 'assistant', 'chatbot']):
            return self._format_rag_response()
        
        # Default response
        return self._format_general_response()

    def _format_skills_response(self) -> str:
        skills = self.profile["skills"]
        return f"""🚀 **Technical Expertise:**

**Languages:** {', '.join(skills['languages'])}
**Backend:** {', '.join(skills['backend'])} - I've built scalable APIs handling 15k+ requests/min
**AI/ML:** {', '.join(skills['ai_ml'])} - Specialized in production-ready ML systems
**Databases:** {', '.join(skills['databases'])} - From SQL to vector databases

I've solved 500+ LeetCode problems and focus on algorithmic excellence in production systems. Want to know about any specific technology or project?"""

    def _format_projects_response(self) -> str:
        return """🎯 **Featured Projects:**

**🌾 Krishak AI** - Agricultural platform with 71.35% disease detection accuracy, helping 1000+ farmers
**🤖 AutoML Platform** - Democratizing ML with 80% faster development for non-technical users  
**🧠 RAG Assistant** - Multi-agent system with 30% faster response times using advanced vector search

Each project combines technical excellence with real-world impact. I focus on production-ready solutions that transform lives! Which project interests you most?"""

    def _format_contact_response(self) -> str:
        contact = self.profile["contact"]
        return f"""📫 **Let's Connect:**

**Email:** {contact['email']}
**GitHub:** {contact['github']}
**LinkedIn:** {contact['linkedin']}
**LeetCode:** {contact['leetcode']}

I respond within 24 hours and love discussing new opportunities, technical challenges, or collaboration ideas. Whether it's AI/ML projects, full-stack development, or mentoring - I'm always excited to help!"""

    def _format_education_response(self) -> str:
        return f"""🎓 **Education & Learning:**

**Current:** {self.profile['education']}
**CGPA:** {self.profile['cgpa']} - Consistently high performance
**Location:** {self.profile['location']}

My learning philosophy: "Learn by building real-world solutions." Every project pushes boundaries and forces deep dives into new technologies. I combine hands-on implementation with algorithmic rigor to create impactful technology!"""

    def _format_availability_response(self) -> str:
        return f"""✅ **{self.profile['availability']}**

**Collaboration Types:**
• AI/ML project development with production deployment
• Full-stack web applications with robust backend architecture  
• Technical consulting and performance optimization
• Mentoring and knowledge sharing

**Response Time:** Within 24 hours
**Timezone:** IST (UTC +5:30)

I bring algorithmic rigor, full-stack expertise, and a proven track record of reducing system latency while improving scalability. Ready to discuss your project!"""

    def _format_krishak_response(self) -> str:
        project = self.profile["projects"]["krishak_ai"]
        return f"""🌾 **{project['name']}**

{project['description']} - This is my flagship agricultural AI project!

**Key Achievements:**
• {project['achievements'][0]} plant disease detection
• {project['achievements'][1]} for real-time queries
• {project['achievements'][2]} for traditional farmers

**Tech Stack:** {', '.join(project['tech'])}
**Impact:** {project['impact']}

The platform combines FastAPI microservices with CNN models, serving both tech-savvy farm owners and traditional farmers through SMS integration. It's transforming farming practices across rural communities!"""

    def _format_automl_response(self) -> str:
        project = self.profile["projects"]["automl_platform"]
        return f"""🤖 **{project['name']}**

{project['description']} - Democratizing machine learning!

**Key Features:**
• {project['achievements'][0]} with interactive visualizations
• {project['achievements'][1]} for faster results
• {project['achievements'][2]} with instructions

**Tech Stack:** {', '.join(project['tech'])}
**Impact:** {project['impact']}

This platform makes ML accessible to everyone, with Flask microservices orchestrating automated ML pipelines. Upload data, get trained models - it's that simple!"""

    def _format_rag_response(self) -> str:
        project = self.profile["projects"]["rag_assistant"]
        return f"""🧠 **{project['name']}**

{project['description']} - Advanced conversational AI!

**Key Features:**
• {project['achievements'][0]} with conversation memory
• {project['achievements'][1]} for state management  
• {project['achievements'][2]} between specialized agents

**Tech Stack:** {', '.join(project['tech'])}
**Impact:** {project['impact']}

This system combines RAG with multi-agent architecture, using FastAPI for async processing and FAISS for optimized vector search. It's like having multiple AI specialists working together!"""

    def _format_general_response(self) -> str:
        return f"""👋 **About {self.profile['name']}**

I'm a {self.profile['role']} from {self.profile['location']}, passionate about building technology that transforms lives!

**Quick Facts:**
• 🎓 {self.profile['education']} (CGPA: {self.profile['cgpa']})
• 💻 500+ LeetCode problems solved
• 🚀 25+ production-ready projects built
• 👥 50+ students mentored

I specialize in AI/ML, full-stack development, and backend architecture. My projects focus on real-world impact - from helping farmers with AI to democratizing machine learning!

What specific aspect would you like to explore? My projects, skills, or collaboration opportunities?"""

# Initialize assistant
assistant = ArkaPortfolioAssistant()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - optimized for low memory usage"""
    try:
        # Convert history to simple format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history[-5:]]  # Keep only last 5 messages
        
        # Generate response
        response = assistant.generate_response(request.message, history)
        
        return ChatResponse(
            response=response,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, I encountered an error. Please try again!")

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Arka AI Portfolio Assistant",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
def root():
    """Root endpoint with basic info"""
    return {
        "service": "Arka AI Portfolio Assistant",
        "description": "Chatbot for Arkaprabha Banerjee's portfolio",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

# Optimized for Render free tier
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for memory efficiency
        access_log=False,  # Disable access logs to save memory
        log_level="warning"  # Minimal logging
    )

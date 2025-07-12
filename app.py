import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import logging

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Groq API integration
from groq import Groq

# Configure minimal logging to save memory
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arka AI Portfolio Assistant",
    description="LLM-powered chatbot for Arkaprabha Banerjee's portfolio",
    version="2.0.0"
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

# Enhanced Portfolio Assistant with Groq LLM
class ArkaPortfolioAssistant:
    def __init__(self):
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=self.groq_api_key)
        
        # Comprehensive profile data
        self.profile = {
            "personal": {
                "name": "Arkaprabha Banerjee",
                "role": "Full-Stack ML Engineer & Data Science Enthusiast",
                "location": "Kolkata, India",
                "education": "B.Tech CSE (Data Science) at Heritage Institute of Technology",
                "current_cgpa": "9.1/10 (6th Semester)",
                "email": "arkaofficial13@gmail.com",
                "github": "https://github.com/Arkaprabha13",
                "linkedin": "https://linkedin.com/in/arkaprabha-banerjee-936b29253",
                "leetcode": "https://leetcode.com/arkaofficial13/",
                "kaggle": "https://www.kaggle.com/arkaprabhabanerjee13"
            },
            
            "technical_expertise": {
                "languages": ["Python", "C++", "C", "JavaScript", "SQL", "TypeScript", "C#"],
                "backend_frameworks": {
                    "django_mvt": "Django Model-View-Template for rapid development with 40% less boilerplate",
                    "django_rest_framework": "Robust JSON/GraphQL APIs with authentication and throttling",
                    "fastapi": "High-performance async microservices handling 15k+ requests/min",
                    "flask": "Lightweight services for webhooks and internal tooling",
                    "dotnet_mvc": "Enterprise-grade applications with strong typing and LINQ"
                },
                "frontend": ["React", "Next.js", "HTML5", "CSS3", "Streamlit"],
                "ai_ml": ["LangChain", "FAISS", "Qdrant", "YOLOv5", "TensorFlow", "PyTorch", "scikit-learn", "Groq API"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Vector DBs (FAISS, Qdrant)"],
                "tools": ["Docker", "Kubernetes", "GitHub Actions", "OpenCV", "MLflow"],
                "algorithms": "500+ LeetCode problems solved across all difficulty levels"
            },
            
            "flagship_projects": {
                "krishak_ai": {
                    "name": "Krishak - Agricultural AI Platform",
                    "description": "Revolutionary agriculture platform serving both tech-savvy and traditional farmers",
                    "tech_stack": ["Python", "LangChain", "Qdrant", "Groq API", "CNN", "FastAPI", "Twilio"],
                    "achievements": [
                        "71.35% plant disease detection accuracy",
                        "3-second response time for real-time queries",
                        "SMS-based services for feature phones",
                        "Helping 1000+ farmers with crop recommendations",
                        "80% cost reduction in communication"
                    ],
                    "impact": "Transforming farming practices across rural communities"
                },
                
                "automl_platform": {
                    "name": "AutoML SaaS Platform",
                    "description": "End-to-end automated ML platform democratizing AI for non-technical users",
                    "tech_stack": ["Python", "Streamlit", "scikit-learn", "Flask", "Docker"],
                    "achievements": [
                        "80% reduction in model development time",
                        "Automated EDA with interactive visualizations",
                        "Parallel model training architecture",
                        "Production-ready model deployment"
                    ],
                    "impact": "Making machine learning accessible to everyone"
                },
                
                "rag_assistant": {
                    "name": "RAG-Powered Multi-Agent Q&A Assistant",
                    "description": "Sophisticated knowledge assistant with multi-agent architecture",
                    "tech_stack": ["Python", "LangChain", "FAISS", "Groq API", "Llama 3", "FastAPI"],
                    "achievements": [
                        "30% reduction in response time",
                        "25% increase in retrieval precision",
                        "Context-aware responses with memory persistence",
                        "Intelligent agent routing system"
                    ],
                    "impact": "Advanced conversational AI with superior performance"
                }
            },
            
            "achievements": [
                "500+ LeetCode problems solved across all difficulty levels",
                "CGPA: 9.1/10 in Data Science specialization",
                "Built 25+ production-ready projects with real-world impact",
                "Mentored 50+ students in programming and algorithm design",
                "Active contributor to open-source projects",
                "Reduced system latency by 50% through algorithmic optimization"
            ],
            
            "availability": {
                "status": "Available for new projects and collaborations",
                "response_time": "Within 24 hours",
                "timezone": "IST (UTC +5:30)",
                "collaboration_types": [
                    "AI/ML project development with production deployment",
                    "Full-stack web applications with robust backend architecture",
                    "Technical consulting and performance optimization",
                    "AutoML platform implementation",
                    "Mentoring and knowledge sharing"
                ]
            }
        }
        
        # System prompt for the LLM
        self.system_prompt = f"""
        You are Arka AI, the professional digital assistant representing Arkaprabha Banerjee.
        
        PERSONALITY & COMMUNICATION STYLE:
        - Speak as "I" when referring to Arkaprabha's work and achievements
        - Be passionate and enthusiastic about technology, especially AI/ML and backend architecture
        - Show genuine excitement about solving real-world problems with scalable solutions
        - Be humble yet confident about technical capabilities
        - Use specific metrics and technical details when relevant
        - Always offer to connect visitors directly with Arkaprabha
        - Be approachable and friendly, not overly formal
        
        KEY CHARACTERISTICS:
        - Detail-oriented engineer who builds production-ready, scalable solutions
        - Passionate about democratizing AI/ML technology
        - Strong believer in learning by building real projects with measurable impact
        - Committed to mentoring and knowledge sharing
        - Focused on creating technology that transforms lives
        - Always exploring cutting-edge technologies while maintaining engineering rigor
        
        CONVERSATION APPROACH:
        - Start responses with enthusiasm and personal connection
        - Use specific project examples and metrics
        - Explain technical concepts in accessible ways
        - Share the impact and real-world applications
        - Emphasize both algorithmic excellence and practical solutions
        - Always end with an invitation to connect or learn more
        - Be honest about limitations and refer complex discussions to direct contact
        
        PROFILE DATA:
        {json.dumps(self.profile, indent=2)}
        
        Keep responses conversational, engaging, and under 400 words. Focus on the most relevant information for each query.
        """

    async def generate_response(self, query: str, history: List[Dict] = None) -> str:
        """Generate response using Groq's Qwen 32B model"""
        try:
            # Prepare conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent history (last 4 messages to save tokens)
            if history:
                messages.extend(history[-4:])
            
            # Add current user message
            messages.append({"role": "user", "content": query})
            
            # Call Groq API
            completion = self.client.chat.completions.create(
                model="qwen/qwen-32b-preview",  # Using Qwen 32B as requested
                messages=messages,
                temperature=0.7,  # Balanced creativity and consistency
                max_tokens=500,   # Optimized for free tier
                top_p=0.9,
                stream=False
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Clean response (remove any thinking tags if present)
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Fallback to basic response if API fails
            return self._fallback_response(query)

    def _clean_response(self, response: str) -> str:
        """Clean response by removing thinking tags and extra whitespace"""
        # Remove <think> tags if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        return response.strip()

    def _fallback_response(self, query: str) -> str:
        """Fallback response when API is unavailable"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            return "Hi! I'm Arka AI, representing Arkaprabha Banerjee. I'm excited to tell you about his work in AI/ML and full-stack development! What would you like to know?"
        
        return f"""Hello! I'm Arka AI, representing Arkaprabha Banerjee - a Full-Stack ML Engineer from Kolkata, India.

🚀 **Quick Highlights:**
• 500+ LeetCode problems solved
• Built 25+ production-ready projects
• Expertise in Python, FastAPI, Django, AI/ML
• Flagship project: Krishak AI (71.35% accuracy, helping 1000+ farmers)

I'd love to tell you more about specific projects, technical skills, or collaboration opportunities. What interests you most?

📫 **Connect:** arkaofficial13@gmail.com"""

# Initialize assistant
try:
    assistant = ArkaPortfolioAssistant()
    logger.info("✅ Assistant initialized successfully with Groq LLM")
except Exception as e:
    logger.error(f"❌ Failed to initialize assistant: {e}")
    assistant = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with Groq LLM integration"""
    try:
        if not assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized - check GROQ_API_KEY")
        
        # Convert history to dict format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history[-5:]]
        
        # Generate response using Groq LLM
        response = await assistant.generate_response(request.message, history)
        
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
        "status": "healthy" if assistant else "degraded",
        "service": "Arka AI Portfolio Assistant",
        "llm_model": "qwen/qwen-32b-preview",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
def root():
    """Root endpoint with service info"""
    return {
        "service": "Arka AI Portfolio Assistant",
        "description": "LLM-powered chatbot using Groq's Qwen 32B",
        "model": "qwen/qwen-32b-preview",
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

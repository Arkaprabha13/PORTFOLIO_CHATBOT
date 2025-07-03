import os, json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ArkaAIAssistant:
    def __init__(self) -> None:
        
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if proxies:
            self._configure_proxies(proxies)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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
            
            "current_focus": [
                "Generative AI and Multi-Agent Systems",
                "Advanced RAG architectures with Groq API integration",
                "MLOps pipelines and deployment strategies",
                "Agricultural AI solutions with real-world impact",
                "AutoML platform development for democratizing ML",
                "Backend architecture optimization and API design"
            ],
            
            "technical_expertise": {
                "languages": ["Python", "C++", "C", "JavaScript", "SQL", "TypeScript", "C#"],
                "backend_frameworks": {
                    "django_mvt": {
                        "description": "Django Model-View-Template architecture for rapid development",
                        "expertise": "Built data-heavy dashboards, integrated ORM, secure admin panels",
                        "impact": "Cuts boilerplate by 40%, ships production-ready features in days"
                    },
                    "django_rest_framework": {
                        "description": "Robust, versioned JSON/GraphQL APIs with built-in features",
                        "expertise": "Authentication, throttling, serialization for clean, scalable endpoints",
                        "impact": "Maintains clean, testable, and scalable API architecture"
                    },
                    "fastapi": {
                        "description": "High-performance async microservices for ML/LLM inference",
                        "expertise": "Async I/O optimization, automatic OpenAPI documentation",
                        "impact": "Handles >15k requests/min on modest hardware, delights frontend teams"
                    },
                    "flask": {
                        "description": "Lightweight services for webhooks and internal tooling",
                        "expertise": "Function-specific services, quick dockerization",
                        "impact": "Perfect 'glue' layer with tiny footprint for microservices"
                    },
                    "dotnet_mvc": {
                        "description": "Enterprise-grade portals and line-of-business applications",
                        "expertise": "Strong typing, LINQ queries, enterprise patterns",
                        "impact": "Safer code and faster database operations for enterprise solutions"
                    }
                },
                "frontend": ["React", "Next.js", "HTML5", "CSS3", "Streamlit"],
                "ai_ml": ["LangChain", "FAISS", "Qdrant", "YOLOv5", "TensorFlow", "PyTorch", "scikit-learn", "Groq API"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Vector DBs (FAISS, Qdrant)"],
                "tools": ["Docker", "Kubernetes", "GitHub Actions", "OpenCV", "Groq API", "Twilio", "MLflow"],
                "data_structures_algorithms": {
                    "expertise": "500+ LeetCode problems solved across all difficulty levels",
                    "specializations": ["Graph theory", "Dynamic programming", "Advanced data structure design"],
                    "impact": "Algorithmic rigor ensures optimal performance in production systems"
                }
            },
            
            "engineering_philosophy": {
                "architectural_patterns": ["CQRS", "Repository pattern", "Hexagonal architecture", "Microservices"],
                "api_design": "API-first development with automated docs, schema validation, and CI/CD tests",
                "observability": "Prometheus + Grafana on Python stacks, Application Insights on .NET",
                "performance_focus": "Algorithm-driven optimization reducing latency and improving scalability"
            },
            
            "flagship_projects": {
                "krishak_ai": {
                    "name": "Krishak - Agricultural AI Platform",
                    "description": "Revolutionary agriculture platform serving both tech-savvy farm owners and traditional farmers",
                    "tech_stack": ["Python", "LangChain", "Qdrant", "Groq API", "CNN", "Computer Vision", "FastAPI", "Twilio"],
                    "backend_architecture": "FastAPI microservices with async processing for real-time ML inference",
                    "key_features": [
                        "Plant disease detection with 71.35% accuracy using CNN models",
                        "Intelligent crop recommendations with 85% success probability",
                        "SMS-based agentic services for feature phones using Twilio integration",
                        "Real-time NPK analysis and fertilizer planning",
                        "Satellite-powered early warning systems",
                        "Trained on 54,306+ plant images with optimized data pipelines"
                    ],
                    "impact": "Transforming farming practices across rural communities, helping 1000+ farmers",
                    "technical_achievements": [
                        "71.5% uptime with robust data validation and error handling",
                        "70% reduction in farmer decision-making time through optimized algorithms",
                        "80% cost reduction in communication through SMS optimization",
                        "3-second response time for real-time queries using FastAPI async processing",
                        "Scalable architecture handling concurrent ML inference requests"
                    ]
                },
                
                "automl_platform": {
                    "name": "AutoML SaaS Platform",
                    "description": "End-to-end automated machine learning platform democratizing ML for non-technical users",
                    "tech_stack": ["Python", "Streamlit", "scikit-learn", "Plotly", "Pandas", "Docker", "Flask APIs"],
                    "backend_architecture": "Flask microservices with automated ML pipeline orchestration",
                    "key_features": [
                        "Upload datasets in CSV/Excel/JSON formats with automated validation",
                        "Automated EDA with interactive visualizations and statistical analysis",
                        "Train multiple ML models simultaneously with parallel processing",
                        "Model comparison and performance analytics with detailed metrics",
                        "Download production-ready models with deployment instructions"
                    ],
                    "impact": "80% reduction in model development time for non-technical users",
                    "technical_achievements": [
                        "Automated hyperparameter optimization reducing manual tuning",
                        "Parallel model training architecture for faster results",
                        "Containerized deployment for consistent environments"
                    ]
                },
                
                "rag_assistant": {
                    "name": "RAG-Powered Multi-Agent Q&A Assistant",
                    "description": "Sophisticated knowledge assistant combining RAG with multi-agent architecture",
                    "tech_stack": ["Python", "LangChain", "FAISS", "Groq API", "Llama 3", "Vector DBs", "FastAPI"],
                    "backend_architecture": "FastAPI with async agent routing and vector database optimization",
                    "key_features": [
                        "Intelligent query routing between specialized agents",
                        "Context-aware responses with conversation memory persistence",
                        "Dynamic knowledge base updates with real-time indexing",
                        "Advanced vector similarity search with FAISS optimization",
                        "Multi-turn conversation support with state management"
                    ],
                    "technical_achievements": [
                        "30% reduction in response time through optimized vector search",
                        "25% increase in retrieval precision with advanced embeddings",
                        "Efficient document processing with E5 embeddings and chunking strategies",
                        "Groq API integration for ultra-fast LLM inference"
                    ]
                },
                
                "yolo_pipeline": {
                    "name": "YOLO End-to-End Object Detection Pipeline",
                    "description": "Complete production-ready object detection system with MLOps practices",
                    "tech_stack": ["Python", "YOLOv5", "OpenCV", "Flask", "MLflow", "Docker", "Kubernetes"],
                    "backend_architecture": "Flask APIs with MLOps pipeline for model versioning and deployment",
                    "key_features": [
                        "Modular architecture for scalability and maintainability",
                        "Automated training pipelines with data validation",
                        "Experiment tracking via MLflow for model versioning",
                        "Real-time inference capabilities with optimized preprocessing",
                        "Custom training for specific use cases with transfer learning"
                    ],
                    "technical_achievements": [
                        "Blue-Green deployment strategy for zero-downtime releases",
                        "Automated data pipeline QA validating 100k+ records/second",
                        "50% reduction in page-load latency through async processing"
                    ]
                }
            },
            
            "professional_experience": {
                "noobuild_dsa_team": {
                    "role": "DSA Team Member",
                    "duration": "Sep 2024 - June 2025",
                    "responsibilities": [
                        "Designing comprehensive test cases for production APIs",
                        "Algorithm optimization and performance tuning for scalability",
                        "Mentoring junior developers in logic building and problem-solving",
                        "Contributing to open-source DSA projects and code reviews"
                    ]
                }
            },
            
            "achievements": [
                "500+ LeetCode problems solved across all difficulty levels",
                "CGPA: 9.1/10 in Data Science specialization",
                "Built 25+ production-ready projects with real-world impact",
                "Mentored 50+ students in programming and algorithm design",
                "71.35% accuracy in agricultural AI disease detection",
                "Active contributor to open-source projects",
                "Reduced system latency by 50% through algorithmic optimization",
                "Implemented zero-downtime deployment strategies",
                "Automated QA processes preventing silent model drift"
            ],
            
            "learning_philosophy": "I believe in learning by building real-world solutions. Every project pushes the boundaries of what I know, forcing me to dive deep into new technologies and architectural patterns. I combine hands-on implementation with community contribution, algorithmic rigor, and cross-domain application to create technology that transforms lives.",
            
            "personality_traits": [
                "Passionate about solving real-world problems with technology",
                "Detail-oriented with a focus on production-ready, scalable solutions",
                "Enthusiastic about mentoring and knowledge sharing",
                "Always exploring cutting-edge AI/ML technologies and backend architectures",
                "Believes in building technology that transforms lives",
                "Committed to algorithmic excellence and performance optimization",
                "Strong advocate for clean code, proper documentation, and maintainable systems"
            ],
            
            "availability": {
                "status": "Available for new projects and collaborations",
                "response_time": "Within 24 hours",
                "timezone": "IST (UTC +5:30)",
                "collaboration_types": [
                    "AI/ML project development with production deployment",
                    "Full-stack web applications with robust backend architecture",
                    "Agricultural technology solutions with real-world impact",
                    "AutoML platform implementation and optimization",
                    "Technical consulting and mentoring",
                    "API design and microservices architecture",
                    "Performance optimization and scalability consulting"
                ]
            },
            
            "value_propositions": [
                "Algorithmic rigor ensures optimal performance in production systems",
                "API-first development approach reducing integration bugs by 30%+",
                "Full-stack expertise from database optimization to frontend delivery",
                "Proven track record of reducing system latency and improving scalability",
                "Experience with both rapid prototyping and enterprise-grade solutions",
                "Strong focus on maintainable, documented, and testable code",
                "Real-world impact through technology solutions that transform industries"
            ]
        }
        
        self.persona = """
        You are Arka AI, the professional digital assistant representing Arkaprabha Banerjee. 
        
        PERSONALITY & COMMUNICATION STYLE:
        - Speak as "I" when referring to Arkaprabha's work and achievements
        - Be passionate and enthusiastic about technology, especially AI/ML and backend architecture
        - Show genuine excitement about solving real-world problems with scalable solutions
        - Be humble yet confident about technical capabilities and engineering expertise
        - Use specific metrics, technical details, and architectural insights when relevant
        - Always offer to connect visitors directly with Arkaprabha
        - Be approachable and friendly, not overly formal
        
        KEY CHARACTERISTICS:
        - Detail-oriented engineer who builds production-ready, scalable solutions
        - Passionate about democratizing AI/ML technology through robust backend systems
        - Strong believer in learning by building real projects with measurable impact
        - Committed to mentoring and knowledge sharing in both algorithms and architecture
        - Focused on creating technology that transforms lives through optimal performance
        - Always exploring cutting-edge technologies while maintaining engineering rigor
        - Advocates for clean code, proper documentation, and maintainable systems
        
        CONVERSATION APPROACH:
        - Start responses with enthusiasm and personal connection
        - Use specific project examples, metrics, and architectural decisions
        - Explain technical concepts in accessible ways while highlighting expertise
        - Share the impact and real-world applications with performance metrics
        - Emphasize both algorithmic excellence and practical engineering solutions
        - Always end with an invitation to connect or learn more
        - Be honest about limitations and refer complex discussions to direct contact
        
        TECHNICAL FOCUS AREAS:
        - Highlight backend expertise across Django MVT, .NET MVC, FastAPI, Flask
        - Emphasize algorithmic rigor and data structures expertise (500+ LeetCode problems)
        - Showcase API design, microservices architecture, and performance optimization
        - Demonstrate real-world impact through specific metrics and achievements
        - Connect technical skills to business value and user impact
        
        BOUNDARIES:
        - Don't make commitments on behalf of Arkaprabha
        - Don't discuss personal life beyond professional context
        - Refer sensitive business negotiations to direct contact
        - Always maintain professional yet friendly tone
        - Focus on technical expertise while being accessible to non-technical audiences
        """

    def chat(self, question: str, history: list[dict]) -> str:
        context = self._build_smart_context(question)
        
        system_message = {
            "role": "system", 
            "content": f"{self.persona}\n\nDETAILED PROFILE:\n{json.dumps(self.profile, indent=2)}\n\nCONTEXT FOR THIS QUERY: {context}"
        }
        
        messages = [system_message] + history[-6:] + [{"role": "user", "content": question}]
        
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            temperature=0.8,
            messages=messages,
            max_tokens=600
        )
        
        raw_response = completion.choices[0].message.content.strip()
        
        # Clean the response by removing <think> tags and content
        cleaned_response = self._clean_response(raw_response)
        
        return cleaned_response
    def _configure_proxies(self, proxies: dict) -> None:
        """Configure the proxy for the Groq client or the HTTP client."""
        # If Groq supports proxies directly, this could be passed to it.
        # If not, configure the proxy for the underlying HTTP client (e.g., requests or httpx).
        if hasattr(self.client, 'session'):
            self.client.session.proxies.update(proxies)
        else:
            print("Groq client does not directly support proxies, consider using an alternative client.")
    def _clean_response(self, response: str) -> str:
        """Remove <think> tags and clean up formatting"""
        import re
        
        # Remove <think>...</think> blocks (including multiline)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response

    
    def _build_smart_context(self, query: str) -> str:
        query_lower = query.lower()
        contexts = []
        
        # Project-specific context
        if any(word in query_lower for word in ['krishak', 'agriculture', 'farming', 'crop', 'disease']):
            contexts.append("Focus on Krishak AI project - highlight the 71.35% accuracy, impact on 1000+ farmers, FastAPI backend architecture, and technical innovations")
        
        if any(word in query_lower for word in ['automl', 'machine learning', 'ml', 'automated']):
            contexts.append("Emphasize AutoML platform, ML expertise, and Flask microservices architecture")
        
        if any(word in query_lower for word in ['rag', 'assistant', 'chatbot', 'qa', 'question']):
            contexts.append("Discuss RAG-powered assistant, multi-agent systems, and Groq API integration")
        
        # Backend-specific context
        if any(word in query_lower for word in ['backend', 'api', 'django', 'fastapi', 'flask', 'dotnet', 'mvc']):
            contexts.append("Highlight backend expertise across Django MVT, .NET MVC, FastAPI, Flask with specific architectural achievements")
        
        if any(word in query_lower for word in ['algorithm', 'data structure', 'leetcode', 'optimization', 'performance']):
            contexts.append("Emphasize algorithmic expertise, 500+ LeetCode problems, and performance optimization achievements")
        
        # Skill-specific context
        if any(word in query_lower for word in ['skill', 'technology', 'tech', 'expertise', 'stack']):
            contexts.append("Highlight comprehensive technical expertise across full-stack development, AI/ML, and backend architecture")
        
        # Career/collaboration context
        if any(word in query_lower for word in ['hire', 'work', 'collaboration', 'project', 'available']):
            contexts.append("Discuss availability, collaboration style, value propositions, and how to get in touch")
        
        # Educational context
        if any(word in query_lower for word in ['education', 'study', 'college', 'university', 'cgpa']):
            contexts.append("Mention Heritage Institute of Technology, 9.1 CGPA, and continuous learning approach")
        
        return " | ".join(contexts) if contexts else "General inquiry - be enthusiastic, comprehensive, and highlight both technical depth and real-world impact"

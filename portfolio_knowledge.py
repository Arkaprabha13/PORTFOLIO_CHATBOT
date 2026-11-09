"""Portfolio knowledge and scope rules shared by the chatbot entrypoints."""

PROJECT_CATALOG = [
    {
        "name": "Portfolio Chatbot",
        "repo": "PORTFOLIO_CHATBOT",
        "url": "https://github.com/Arkaprabha13/PORTFOLIO_CHATBOT",
        "summary": "FastAPI and Groq-powered portfolio assistant for answering questions about Arkaprabha's work.",
        "stack": ["Python", "FastAPI", "Groq", "Qwen"],
        "highlights": ["Profile-aware responses", "Project knowledge catalog", "Strict portfolio scope policy"],
    },
    {
        "name": "Aegis Sandbox",
        "repo": "aegis-sandbox",
        "url": "https://github.com/Arkaprabha13/aegis-sandbox",
        "summary": "Autonomous red-team platform for testing LLM agents inside an isolated Docker sandbox.",
        "stack": ["Python", "LangGraph", "FastAPI", "Docker", "SQLite", "OWASP LLM Top 10"],
        "highlights": [
            "Uses scripted and adaptive attackers against a sandboxed target agent",
            "Detects real tool-side effects through a mock API instead of trusting model replies",
            "Includes regression testing and OWASP mapping for discovered vulnerabilities",
        ],
    },
    {
        "name": "RAG-Powered Multi-Agent Q&A Assistant",
        "repo": "Rag_powered_assistant-",
        "url": "https://github.com/Arkaprabha13/Rag_powered_assistant-",
        "summary": "Document-grounded Q&A assistant with specialized RAG, calculator, and dictionary tools.",
        "stack": ["Python", "FAISS", "E5 embeddings", "LangChain", "Groq", "Streamlit", "FastAPI"],
        "highlights": [
            "Processes PDFs and text into 1,000-character chunks with 200-character overlap",
            "Retrieves the top three relevant chunks for grounded answers",
            "Provides document ingestion and REST API endpoints",
        ],
    },
    {
        "name": "ArXiv Intelligence Agent",
        "repo": "ArxivAgent",
        "url": "https://github.com/Arkaprabha13/ArxivAgent",
        "summary": "AI research assistant that harvests, scores, summarizes, and searches AI/ML papers.",
        "stack": ["FastAPI", "Gemini", "Groq", "Qdrant", "SQLite", "APScheduler"],
        "highlights": ["Daily paper harvesting", "Vector search and embeddings", "Breakthrough, solid, and niche scoring"],
    },
    {
        "name": "JobBuddy / JobFlow AI",
        "repo": "JobBuddy",
        "url": "https://github.com/Arkaprabha13/JobBuddy",
        "summary": "Autonomous job-application suite for resume tailoring, job analysis, and application workflows.",
        "stack": ["Python", "LangGraph", "FastAPI", "React", "Supabase", "Groq", "Gemini", "Playwright"],
        "highlights": ["ATS resume scoring", "Startup intelligence", "Human-in-the-loop agentic workflow", "PDF generation"],
    },
    {
        "name": "EURON Legal Workflow Agent",
        "repo": "Agentx",
        "url": "https://github.com/Arkaprabha13/Agentx",
        "summary": "Multi-agent legal workflow platform for contract review, risk, compliance, redlining, and cross-referencing.",
        "stack": ["Python", "FastAPI", "Groq", "Supabase", "PostgreSQL"],
        "highlights": ["Specialized legal agents", "Persistent authentication and storage", "Unit, integration, and API tests"],
    },
    {
        "name": "Gemini KYC Data Extractor Pro",
        "repo": "KYC",
        "url": "https://github.com/Arkaprabha13/KYC",
        "summary": "Streamlit KYC document extraction and review application.",
        "stack": ["Python", "Gemini", "Streamlit", "Excel", "JSON"],
        "highlights": ["Indian PAN and Aadhaar validation", "Confidence scoring", "Editable review flow", "Multiple export formats"],
    },
    {
        "name": "Krishak Agricultural RAG System",
        "repo": "Krishak_API",
        "url": "https://github.com/Arkaprabha13/Krishak_API",
        "summary": "Agricultural data ingestion, vector search, and LLM recommendation API.",
        "stack": ["Python", "Qdrant", "Groq", "FastAPI", "Pandas"],
        "highlights": ["Supports CSV, XLS, and PDF agricultural datasets", "Crop and sensor-data retrieval", "Configurable processing pipeline"],
    },
    {
        "name": "Blog Empire",
        "repo": "BLOG_AGENT",
        "url": "https://github.com/Arkaprabha13/BLOG_AGENT",
        "summary": "Autonomous AI blogging platform with content and news-intelligence workflows.",
        "stack": ["Python", "LangGraph", "Groq", "Telegram", "FastAPI"],
        "highlights": ["News-to-blog generation pipeline", "Daily news intelligence", "Telegram bot and web/API interfaces"],
    },
    {
        "name": "AI Email Agent",
        "repo": "Email_agent_backend",
        "url": "https://github.com/Arkaprabha13/Email_agent_backend",
        "summary": "LangGraph email workflow that classifies, prioritizes, summarizes, drafts, and routes messages for review.",
        "stack": ["Python", "LangGraph", "FastAPI", "Groq", "OAuth 2.0", "JWT", "Telegram"],
        "highlights": ["Work, personal, finance, spam, and other classification", "Calendar and notes integrations", "Encrypted OAuth token storage and human review"],
    },
    {
        "name": "AI Email Agent Frontend",
        "repo": "Email_agent_frontend",
        "url": "https://github.com/Arkaprabha13/Email_agent_frontend",
        "summary": "JavaScript frontend repository paired with the AI Email Agent backend.",
        "stack": ["JavaScript"],
        "highlights": ["Frontend repository for the email automation project"],
    },
    {
        "name": "New Portfolio",
        "repo": "NEW_PORTFOLIO",
        "url": "https://github.com/Arkaprabha13/NEW_PORTFOLIO",
        "summary": "TypeScript portfolio project.",
        "stack": ["TypeScript"],
        "highlights": ["Portfolio-focused frontend repository"],
    },
    {
        "name": "Debolina Movers and Packers",
        "repo": "Debolina_movers_and_packers",
        "url": "https://github.com/Arkaprabha13/Debolina_movers_and_packers",
        "summary": "TypeScript web project for a movers-and-packers business.",
        "stack": ["TypeScript"],
        "highlights": ["Business website project"],
    },
    {
        "name": "Codeshetra Django",
        "repo": "codeshetra-django",
        "url": "https://github.com/Arkaprabha13/codeshetra-django",
        "summary": "Mock-interview platform with email authentication, WebRTC video interviews, and Razorpay.",
        "stack": ["Django", "HTML", "CSS", "JavaScript", "WebRTC", "Razorpay"],
        "highlights": ["Email confirmation authentication", "Separate secure video-chat environment", "Real-time interviewer and interviewee sessions"],
    },
    {
        "name": "YOLO End-to-End Pipeline",
        "repo": "YOLO_END_TO_END",
        "url": "https://github.com/Arkaprabha13/YOLO_END_TO_END",
        "summary": "Production-oriented object-detection pipeline with MLOps practices.",
        "stack": ["Python", "YOLOv5", "OpenCV", "Flask", "MLflow", "Docker", "Kubernetes"],
        "highlights": ["Automated training and validation", "Experiment tracking", "Real-time inference and transfer learning"],
    },
    {
        "name": "AutoML Platform",
        "repo": "AUTO_ML",
        "url": "https://github.com/Arkaprabha13/AUTO_ML",
        "summary": "Automated machine-learning workflow for dataset analysis, model training, and comparison.",
        "stack": ["Python", "scikit-learn", "Pandas", "Plotly", "Streamlit"],
        "highlights": ["Automated EDA", "Model comparison", "Interactive visualizations", "Production-model export"],
    },
]

IN_SCOPE_TERMS = {
    "arka", "arkaprabha", "portfolio", "project", "projects", "repository", "repo",
    "github", "work", "experience", "education", "college", "cgpa", "skill", "skills",
    "technology", "technologies", "tech", "stack", "backend", "frontend", "api",
    "python", "django", "fastapi", "flask", "react", "typescript", "javascript",
    "machine learning", "ai", "ml", "rag", "agent", "chatbot", "automl", "agriculture",
    "krishak", "arxiv", "jobbuddy", "e g", "euron", "kyc", "aegis", "email", "blog",
    "leetcode", "hire", "hiring", "collaborate", "collaboration", "contact", "email",
    "linkedin", "availability", "resume", "achievement", "achievement",
}

GREETING_TERMS = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}


def is_in_scope(question: str) -> bool:
    """Return whether a question is about Arkaprabha's professional portfolio."""
    normalized = " ".join(question.lower().split())
    if not normalized:
        return False
    if normalized in GREETING_TERMS:
        return True
    return any(term in normalized for term in IN_SCOPE_TERMS)


def scope_response() -> str:
    return (
        "I can help with Arkaprabha Banerjee's portfolio, projects, repositories, "
        "technical skills, education, experience, availability, and contact details. "
        "I don't have reliable context for that unrelated topic."
    )

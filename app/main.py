from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.routers import chat, health, admin
from app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage de l'API ACL Chatbot...")
    logger.info("📚 Chargement de la knowledge base...")
    rag_service.initialize(settings.KNOWLEDGE_BASE_DIR)
    logger.info("✅ Knowledge base chargée!")
    app.state.rag_service = rag_service
    yield
    logger.info("🛑 Arrêt de l'API...")


app = FastAPI(
    title="AfricTivistes CitizenLab - Chatbot API",
    description="API conversationnelle intelligente d'ACL propulsée par Groq et RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Bienvenue sur l'API Chatbot AfricTivistes CitizenLab",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }
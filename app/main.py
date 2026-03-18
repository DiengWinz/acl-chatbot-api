from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import httpx
from pathlib import Path

from app.config import settings
from app.routers import chat, health, admin
from app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKOFFICE_URL      = os.environ.get("BACKOFFICE_URL", "https://acl-backoffice-api.onrender.com")
BACKOFFICE_USERNAME = os.environ.get("BACKOFFICE_USERNAME", "admin")
BACKOFFICE_PASSWORD = os.environ.get("BACKOFFICE_PASSWORD", "Farimata55@")

rag_service = RAGService()


async def get_backoffice_token() -> str:
    """Récupère un token JWT du back-office."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{BACKOFFICE_URL}/api/auth/token/",
                json={
                    "username": BACKOFFICE_USERNAME,
                    "password": BACKOFFICE_PASSWORD,
                },
                timeout=10,
            )
            res.raise_for_status()
            return res.json().get("access", "")
    except Exception as e:
        logger.warning(f"⚠️ Impossible d'obtenir le token back-office : {e}")
        return ""


async def sync_knowledge_from_backoffice():
    """
    Au démarrage, télécharge tous les fichiers uploadés via le back-office
    et les ajoute dans knowledge_base/ pour que le RAGService les charge.
    """
    token = await get_backoffice_token()
    if not token:
        logger.warning("⚠️ Sync KB ignorée — pas de token back-office")
        return 0

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{BACKOFFICE_URL}/api/knowledge/",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            files = data.get("results", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.warning(f"⚠️ Impossible de récupérer la liste KB : {e}")
        return 0

    # Filtre uniquement les fichiers avec status=ready et une URL Cloudinary
    ready_files = [
        f for f in files
        if f.get("status") == "ready" and f.get("file_url")
    ]

    if not ready_files:
        logger.info("ℹ️ Aucun fichier back-office à synchroniser")
        return 0

    downloaded = 0
    async with httpx.AsyncClient() as client:
        for doc in ready_files:
            file_url  = doc.get("file_url", "")
            file_name = doc.get("name", "")
            folder    = doc.get("folder", "ACL_Sn")

            if not file_url or not file_name:
                continue

            dest_dir  = Path(settings.KNOWLEDGE_BASE_DIR) / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / file_name

            # Ne re-télécharge pas si déjà présent (fichiers du repo)
            if dest_path.exists():
                logger.info(f"  ⏭ Déjà présent : {file_name}")
                continue

            try:
                response = await client.get(file_url, timeout=30)
                response.raise_for_status()
                dest_path.write_bytes(response.content)
                downloaded += 1
                logger.info(f"  ✅ Téléchargé : {file_name} → {folder}/")
            except Exception as e:
                logger.error(f"  ❌ Erreur téléchargement {file_name} : {e}")

    return downloaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage de l'API ACL Chatbot...")

    # 1. Synchronise les fichiers uploadés depuis le back-office
    logger.info("🔄 Synchronisation Knowledge Base depuis le back-office...")
    synced = await sync_knowledge_from_backoffice()
    logger.info(f"  → {synced} nouveau(x) fichier(s) synchronisé(s)")

    # 2. Charge la KB complète (fichiers repo + fichiers synchronisés)
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
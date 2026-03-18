from fastapi import APIRouter, Request, Depends, HTTPException
from app.auth import verify_api_key
from app.services.session_manager import session_manager
from app.schemas import AdminStatsResponse
from pathlib import Path
import httpx
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "knowledge_base")


@router.get("/stats", response_model=AdminStatsResponse, summary="Statistiques de l'API")
async def get_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    rag_service = request.app.state.rag_service
    session_stats = session_manager.get_stats()
    return AdminStatsResponse(
        total_sessions=session_stats["total_sessions"],
        active_sessions=session_stats["active_sessions"],
        total_messages=session_stats["total_messages"],
        knowledge_base_stats=rag_service.get_stats()
    )


@router.post("/cleanup", summary="Nettoyer les sessions expirées")
async def cleanup_sessions(api_key: str = Depends(verify_api_key)):
    count = session_manager.cleanup_expired_sessions()
    return {"message": f"{count} sessions expirées supprimées"}


@router.get("/knowledge-base", summary="Détails de la knowledge base")
async def get_kb_details(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    rag_service = request.app.state.rag_service
    stats = rag_service.get_stats()
    return {
        "total_chunks": stats.get("total_chunks", 0),
        "files_loaded": stats.get("files_loaded", 0),
        "file_details": stats.get("file_details", {}),
        "status": "loaded" if rag_service.is_initialized else "not_loaded"
    }


@router.post("/reload-knowledge", summary="Télécharge un fichier et recharge la knowledge base")
async def reload_knowledge(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Appelé par le back-office après chaque upload de document.
    Body JSON attendu :
    {
        "file_url": "https://res.cloudinary.com/...",
        "file_name": "mon_document.pdf",
        "folder": "ACL_Sn"
    }
    """
    body = await request.json()
    file_url  = body.get("file_url", "")
    file_name = body.get("file_name", "")
    folder    = body.get("folder", "ACL_Sn")

    if not file_url or not file_name:
        raise HTTPException(status_code=400, detail="file_url et file_name sont requis")

    # Crée le dossier de destination
    dest_dir = Path(KNOWLEDGE_BASE_DIR) / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_name

    # Télécharge le fichier depuis Cloudinary
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
        logger.info(f"✅ Fichier téléchargé : {dest_path}")
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur téléchargement : {str(e)}")

    # Recharge le RAGService avec tous les fichiers
    rag_service = request.app.state.rag_service
    rag_service.chunks = []
    rag_service.is_initialized = False
    rag_service.initialize(KNOWLEDGE_BASE_DIR)

    stats = rag_service.get_stats()
    logger.info(f"🔄 Knowledge base rechargée : {stats.get('total_chunks', 0)} chunks")

    return {
        "success":      True,
        "file_saved":   str(dest_path),
        "total_chunks": stats.get("total_chunks", 0),
        "files_loaded": stats.get("files_loaded", 0),
    }


@router.delete("/knowledge-base/{file_name}", summary="Supprimer un fichier de la knowledge base")
async def delete_knowledge_file(
    file_name: str,
    request: Request,
    folder: str = "ACL_Sn",
    api_key: str = Depends(verify_api_key)
):
    """Supprime un fichier local et recharge la knowledge base."""
    file_path = Path(KNOWLEDGE_BASE_DIR) / folder / file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{file_name}' introuvable")

    file_path.unlink()
    logger.info(f"🗑 Fichier supprimé : {file_path}")

    # Recharge
    rag_service = request.app.state.rag_service
    rag_service.chunks = []
    rag_service.is_initialized = False
    rag_service.initialize(KNOWLEDGE_BASE_DIR)

    stats = rag_service.get_stats()
    return {
        "success":      True,
        "deleted":      file_name,
        "total_chunks": stats.get("total_chunks", 0),
    }
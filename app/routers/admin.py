from fastapi import APIRouter, Request, Depends
from app.auth import verify_api_key
from app.services.session_manager import session_manager
from app.schemas import AdminStatsResponse

router = APIRouter()


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
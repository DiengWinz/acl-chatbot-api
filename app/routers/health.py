from fastapi import APIRouter, Request
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Vérifier l'état de l'API")
async def health_check(request: Request):
    rag_service = getattr(request.app.state, "rag_service", None)

    kb_loaded = False
    total_chunks = 0

    if rag_service and rag_service.is_initialized:
        kb_loaded = True
        total_chunks = rag_service.stats.get("total_chunks", 0)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        knowledge_base_loaded=kb_loaded,
        total_chunks=total_chunks,
        environment="development"
    )
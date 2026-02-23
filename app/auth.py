from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key not in settings.api_keys_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "API Key invalide",
                "message": "Fournissez une API Key valide dans le header 'X-API-Key'",
                "contact": "Contactez l'équipe ACL pour obtenir votre clé"
            }
        )
    return api_key
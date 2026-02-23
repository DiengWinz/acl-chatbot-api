from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # === GROQ ===
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # === API SECURITY ===
    API_KEYS: str = "acl-dev-key-2024"
    
    # === RAG ===
    KNOWLEDGE_BASE_DIR: str = "knowledge_base"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    # === SESSION ===
    MAX_HISTORY_LENGTH: int = 20
    SESSION_TTL_MINUTES: int = 60
    
    # === APP ===
    ENVIRONMENT: str = "development"

    @property
    def api_keys_list(self) -> List[str]:
        return [k.strip() for k in self.API_KEYS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
"""
Centralized configuration, loaded once from environment variables / .env.

Every other module imports `settings` from here instead of calling
os.getenv() scattered around the codebase.
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env before pydantic-settings reads the environment, so this works
# whether uvicorn was launched from the project root or elsewhere.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Chunking ---
    chunk_target_tokens: int = 3000
    chunk_gap_seconds: float = 2.5

    # --- Limits ---
    max_video_duration_seconds: int = 14400  # 4 hours

    # --- Whisper fallback ---
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"

    # --- Storage ---
    output_dir: str = "outputs"
    jobs_dir: str = "jobs_store"

    @property
    def output_path(self) -> Path:
        p = BASE_DIR / self.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def jobs_path(self) -> Path:
        p = BASE_DIR / self.jobs_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()


def require_gemini_key() -> str:
    """Raise a clear error early instead of a confusing 401 from Google later."""
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return settings.gemini_api_key

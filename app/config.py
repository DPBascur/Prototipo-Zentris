from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    memory_file: Path
    profile_file: Path
    token: str
    ollama_url: str
    ollama_tags_url: str
    allowed_chat_id: str
    fast_model: str
    balanced_model: str
    reasoning_model: str
    vision_model: str
    max_history: int
    request_timeout: int
    stream_update_interval: float
    min_edit_delta: int
    telegram_msg_limit: int


def load_settings() -> Settings:
    load_dotenv(override=True)

    base_dir = Path(__file__).resolve().parent.parent

    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
    ollama_tags_url = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags").strip()
    allowed_chat_id = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "").strip()

    if not token:
        raise RuntimeError("Falta TELEGRAM_TOKEN en .env")
    if not allowed_chat_id:
        raise RuntimeError("Falta TELEGRAM_ALLOWED_CHAT_ID en .env")
    if not allowed_chat_id.lstrip("-").isdigit():
        raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID debe ser numerico")

    fast_model = os.getenv("FAST_MODEL", "phi3").strip()
    balanced_model = os.getenv("BALANCED_MODEL", "mistral").strip()

    return Settings(
        base_dir=base_dir,
        memory_file=base_dir / "memory.json",
        profile_file=base_dir / "profile.json",
        token=token,
        ollama_url=ollama_url,
        ollama_tags_url=ollama_tags_url,
        allowed_chat_id=allowed_chat_id,
        fast_model=fast_model,
        balanced_model=balanced_model,
        reasoning_model=os.getenv("REASONING_MODEL", balanced_model).strip(),
        vision_model=os.getenv("VISION_MODEL", "llava").strip(),
        max_history=int(os.getenv("MAX_HISTORY", "1200")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        stream_update_interval=float(os.getenv("STREAM_UPDATE_INTERVAL", "0.9")),
        min_edit_delta=int(os.getenv("MIN_EDIT_DELTA", "20")),
        telegram_msg_limit=3900,
    )

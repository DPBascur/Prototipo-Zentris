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
    app_mode: str
    voice_record_seconds: int
    voice_sample_rate: int
    voice_model_size: str
    voice_language: str
    voice_input_device: int | None
    voice_tts_enabled: bool
    voice_tts_rate: int
    voice_tts_provider: str
    azure_speech_key: str
    azure_speech_region: str
    azure_speech_voice: str
    voice_noise_calibration_seconds: float
    voice_noise_gate_multiplier: float
    voice_min_speech_seconds: float
    voice_silence_timeout_seconds: float
    voice_max_record_seconds: float
    voice_stt_beam_size: int
    voice_stt_best_of: int
    voice_stt_temperature: float
    voice_stt_no_speech_threshold: float
    voice_stt_logprob_threshold: float
    voice_vad_min_silence_ms: int
    voice_stt_use_vad_filter: bool
    voice_force_fast_model: bool
    voice_profile_extract_llm: bool
    force_gpu: bool
    stt_device: str
    embedding_device: str


def load_settings() -> Settings:
    load_dotenv(override=True)

    base_dir = Path(__file__).resolve().parent.parent

    app_mode = os.getenv("APP_MODE", "telegram").strip().lower()
    if app_mode not in {"telegram", "local", "hybrid"}:
        raise RuntimeError("APP_MODE debe ser: telegram, local o hybrid")

    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
    ollama_tags_url = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags").strip()
    allowed_chat_id = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "").strip() or "local-user"

    if app_mode in {"telegram", "hybrid"}:
        if not token:
            raise RuntimeError("Falta TELEGRAM_TOKEN en .env")
        if not allowed_chat_id:
            raise RuntimeError("Falta TELEGRAM_ALLOWED_CHAT_ID en .env")
        if not allowed_chat_id.lstrip("-").isdigit():
            raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID debe ser numerico")

    fast_model = os.getenv("FAST_MODEL", "phi3").strip()
    balanced_model = os.getenv("BALANCED_MODEL", "mistral").strip()
    voice_input_device_raw = os.getenv("VOICE_INPUT_DEVICE", "").strip()
    voice_input_device = int(voice_input_device_raw) if voice_input_device_raw else None

    voice_tts_enabled = os.getenv("VOICE_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "si"}
    voice_tts_provider = os.getenv("VOICE_TTS_PROVIDER", "local").strip().lower()
    if voice_tts_provider not in {"local", "azure", "none"}:
        raise RuntimeError("VOICE_TTS_PROVIDER debe ser: local, azure o none")

    voice_force_fast_model = os.getenv("VOICE_FORCE_FAST_MODEL", "true").strip().lower() in {"1", "true", "yes", "si"}
    voice_profile_extract_llm = os.getenv("VOICE_PROFILE_EXTRACT_LLM", "false").strip().lower() in {"1", "true", "yes", "si"}
    force_gpu = os.getenv("FORCE_GPU", "false").strip().lower() in {"1", "true", "yes", "si"}

    if force_gpu:
        stt_device = "cuda"
        embedding_device = "cuda"
    else:
        stt_device = os.getenv("STT_DEVICE", "cpu").strip().lower()
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()

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
        app_mode=app_mode,
        voice_record_seconds=int(os.getenv("VOICE_RECORD_SECONDS", "5")),
        voice_sample_rate=int(os.getenv("VOICE_SAMPLE_RATE", "16000")),
        voice_model_size=os.getenv("VOICE_MODEL_SIZE", "base").strip(),
        voice_language=os.getenv("VOICE_LANGUAGE", "es").strip(),
        voice_input_device=voice_input_device,
        voice_tts_enabled=voice_tts_enabled,
        voice_tts_rate=int(os.getenv("VOICE_TTS_RATE", "170")),
        voice_tts_provider=voice_tts_provider,
        azure_speech_key=os.getenv("AZURE_SPEECH_KEY", "").strip(),
        azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "").strip(),
        azure_speech_voice=os.getenv("AZURE_SPEECH_VOICE", "es-MX-JorgeNeural").strip(),
        voice_noise_calibration_seconds=float(os.getenv("VOICE_NOISE_CALIBRATION_SECONDS", "1.0")),
        voice_noise_gate_multiplier=float(os.getenv("VOICE_NOISE_GATE_MULTIPLIER", "2.0")),
        voice_min_speech_seconds=float(os.getenv("VOICE_MIN_SPEECH_SECONDS", "0.5")),
        voice_silence_timeout_seconds=float(os.getenv("VOICE_SILENCE_TIMEOUT_SECONDS", "1.0")),
        voice_max_record_seconds=float(os.getenv("VOICE_MAX_RECORD_SECONDS", "8.0")),
        voice_stt_beam_size=int(os.getenv("VOICE_STT_BEAM_SIZE", "6")),
        voice_stt_best_of=int(os.getenv("VOICE_STT_BEST_OF", "4")),
        voice_stt_temperature=float(os.getenv("VOICE_STT_TEMPERATURE", "0.0")),
        voice_stt_no_speech_threshold=float(os.getenv("VOICE_STT_NO_SPEECH_THRESHOLD", "0.6")),
        voice_stt_logprob_threshold=float(os.getenv("VOICE_STT_LOGPROB_THRESHOLD", "-1.2")),
        voice_vad_min_silence_ms=int(os.getenv("VOICE_VAD_MIN_SILENCE_MS", "500")),
        voice_stt_use_vad_filter=os.getenv("VOICE_STT_USE_VAD_FILTER", "false").strip().lower() in {"1", "true", "yes", "si"},
        voice_force_fast_model=voice_force_fast_model,
        voice_profile_extract_llm=voice_profile_extract_llm,
        force_gpu=force_gpu,
        stt_device=stt_device,
        embedding_device=embedding_device,
    )

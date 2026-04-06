import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from app.application.assistant_core import AssistantCore
from app.config import load_settings
from app.infrastructure.ollama_client import OllamaClient
from app.infrastructure.vector_memory import VectorMemory
from app.presentation.telegram_handlers import TelegramHandlers
from app.presentation.voice_runner import LocalVoiceRunner


def _validate_gpu_if_forced(settings) -> None:
    if not settings.force_gpu:
        return

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "FORCE_GPU=true requiere torch instalado para validar CUDA. "
            "Instala dependencias GPU y vuelve a intentar."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FORCE_GPU=true pero CUDA no esta disponible. "
            "Configura drivers/CUDA o usa FORCE_GPU=false."
        )


def create_runtime():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("zentris")

    settings = load_settings()
    _validate_gpu_if_forced(settings)
    ollama_client = OllamaClient(settings, logger)
    ollama_client.validate_models()

    vector_memory = VectorMemory(settings, logger)
    core = AssistantCore(settings, logger, ollama_client, vector_memory)
    app = None
    voice_runner = None

    if settings.app_mode in {"telegram", "hybrid"}:
        handlers = TelegramHandlers(settings, logger, core, ollama_client)
        app = ApplicationBuilder().token(settings.token).build()
        app.add_handler(CommandHandler("perfil", handlers.handle_profile_command))
        app.add_handler(CommandHandler("editar_perfil", handlers.handle_edit_profile_command))
        app.add_handler(CommandHandler("estilo", handlers.handle_style_command))
        app.add_handler(CommandHandler("emojis", handlers.handle_emojis_command))
        app.add_handler(CommandHandler("chileno", handlers.handle_chilean_command))
        app.add_handler(CommandHandler("borrar_perfil", handlers.handle_clear_profile_command))
        app.add_handler(CommandHandler("borrar_memoria", handlers.handle_clear_memory_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
        app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))

    if settings.app_mode in {"local", "hybrid"}:
        voice_runner = LocalVoiceRunner(settings, logger, core, ollama_client)

    return settings, app, voice_runner

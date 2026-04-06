import logging

from telegram.ext import ApplicationBuilder, MessageHandler, filters

from app.application.assistant_core import AssistantCore
from app.config import load_settings
from app.infrastructure.ollama_client import OllamaClient
from app.infrastructure.vector_memory import VectorMemory
from app.presentation.telegram_handlers import TelegramHandlers


def create_app():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("zentris")

    settings = load_settings()
    ollama_client = OllamaClient(settings, logger)
    ollama_client.validate_models()

    vector_memory = VectorMemory(logger)
    core = AssistantCore(settings, logger, ollama_client, vector_memory)
    handlers = TelegramHandlers(settings, logger, core, ollama_client)

    app = ApplicationBuilder().token(settings.token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))

    return app

import asyncio
import base64
import time

from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from app.infrastructure.language import enforce_spanish


class TelegramHandlers:
    def __init__(self, settings, logger, core, ollama_client):
        self.settings = settings
        self.logger = logger
        self.core = core
        self.ollama = ollama_client

    def is_authorized_private_chat(self, update) -> bool:
        if not update.message:
            return False
        chat = update.effective_chat
        if not chat or chat.type != "private":
            return False

        user_id = str(update.message.chat_id)
        if user_id != self.settings.allowed_chat_id:
            self.logger.info(
                "Mensaje ignorado por chat_id no autorizado: %s (esperado: %s)",
                user_id,
                self.settings.allowed_chat_id,
            )
            return False
        return True

    async def handle_message(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        if not self.is_authorized_private_chat(update):
            return

        chat = update.effective_chat
        user_id = str(update.message.chat_id)
        user_message = update.message.text.strip()

        previous_task = self.core.active_generation_tasks.get(user_id)
        current_task = asyncio.current_task()
        if previous_task and previous_task is not current_task and not previous_task.done():
            previous_task.cancel()

        if current_task is not None:
            self.core.active_generation_tasks[user_id] = current_task

        quick = self.core.fast_response(user_message)
        if quick:
            await update.message.reply_text(quick)
            return

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        history = self.core.user_memory.get(user_id, "")
        profile = self.core.user_profile.get(user_id, {})
        relevant_memory = ""
        if len(user_message.split()) > 5:
            relevant_memory = self.core.vector_memory.get_relevant_memories(user_id, user_message)

        prompt = self.core.build_text_prompt(user_message, profile, relevant_memory, history)
        model = self.core.choose_model(user_message)

        msg = await update.message.reply_text("Pensando...")
        allow_live_stream = bool(profile)
        final_text = ""
        last_update_time = time.monotonic()
        last_published_text = ""

        try:
            try:
                async for chunk in self.ollama.stream_text(prompt, model):
                    final_text = chunk
                    if not allow_live_stream:
                        continue

                    enough_time = (time.monotonic() - last_update_time) >= self.settings.stream_update_interval
                    enough_delta = (len(final_text) - len(last_published_text)) >= self.settings.min_edit_delta
                    if not (enough_time and enough_delta):
                        continue

                    last_update_time = time.monotonic()
                    last_published_text = final_text
                    try:
                        await msg.edit_text(final_text[-self.settings.telegram_msg_limit :])
                    except BadRequest as exc:
                        if "message is not modified" not in str(exc).lower():
                            self.logger.warning("Error al editar mensaje en streaming: %s", exc)
            except Exception as stream_exc:
                if model != self.settings.balanced_model:
                    self.logger.warning(
                        "Fallo modelo %s, fallback a %s: %s",
                        model,
                        self.settings.balanced_model,
                        stream_exc,
                    )
                    final_text = ""
                    async for chunk in self.ollama.stream_text(prompt, self.settings.balanced_model):
                        final_text = chunk
                else:
                    raise

            final_text = self.core.sanitize_reply(profile, final_text.strip())
            final_text = await enforce_spanish(
                final_text,
                self.settings.balanced_model,
                self.ollama.generate_text,
                self.logger,
            )
            if not final_text:
                final_text = "No pude responder."

            parts = self.core.chunk_text(final_text)
            await msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)

        except asyncio.CancelledError:
            self.logger.info("Generacion cancelada para chat %s", user_id)
            return
        except Exception as exc:
            self.logger.exception("Error durante la generacion de respuesta")
            await msg.edit_text(f"Error IA: {exc}")
            return
        finally:
            running = self.core.active_generation_tasks.get(user_id)
            if running is asyncio.current_task():
                self.core.active_generation_tasks.pop(user_id, None)

        self.core.persist_text_interaction(user_id, user_message, final_text)

        if len(user_message.split()) > 5:
            self.core.vector_memory.save_memory(user_id, user_message)

        context.application.create_task(self.core.extract_user_info_async(user_id, user_message))

    async def handle_photo(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.photo:
            return
        if not self.is_authorized_private_chat(update):
            return

        chat = update.effective_chat
        user_id = str(update.message.chat_id)
        caption = (update.message.caption or "").strip()

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        status_msg = await update.message.reply_text("Analizando imagen...")

        try:
            photo = update.message.photo[-1]
            photo_file = await photo.get_file()
            image_bytes = await photo_file.download_as_bytearray()
            image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")

            if not self.settings.vision_model:
                await status_msg.edit_text("No hay modelo de vision configurado. Define VISION_MODEL en .env.")
                return

            if self.settings.vision_model not in self.ollama.available_models:
                await status_msg.edit_text(
                    f"El modelo de vision '{self.settings.vision_model}' no esta instalado en Ollama. "
                    "Instalalo con: ollama pull llava (o cambia VISION_MODEL en .env)."
                )
                return

            prompt = self.core.build_vision_prompt(caption)
            reply = await self.ollama.generate_vision_text(prompt, image_b64, self.settings.vision_model)
            reply = await enforce_spanish(
                reply,
                self.settings.balanced_model,
                self.ollama.generate_text,
                self.logger,
            )
            if not reply:
                reply = "No pude analizar la imagen con el modelo actual."

            parts = self.core.chunk_text(reply)
            await status_msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)

            self.core.persist_image_interaction(user_id, caption, reply)

        except Exception as exc:
            self.logger.exception("Error procesando imagen")
            await status_msg.edit_text(f"No pude analizar la imagen: {exc}")

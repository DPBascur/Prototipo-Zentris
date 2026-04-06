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

    async def _safe_edit_text(self, message, text: str) -> None:
        try:
            await message.edit_text(text)
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    async def handle_profile_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        summary = self.core.get_profile_summary(user_id)
        await update.message.reply_text(summary)

    async def handle_edit_profile_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        raw = " ".join(context.args).strip() if context.args else ""
        if not raw:
            await update.message.reply_text(
                "Uso: /editar_perfil campo=valor; campo2=valor2. "
                "Campos: nombre, edad, city, nationality, idioma_preferido, objetivo, rol_creador, studies, intereses, gato"
            )
            return

        updates = self.core.parse_profile_edit_text(raw)
        if not updates:
            await update.message.reply_text("No detecte cambios validos.")
            return

        self.core.upsert_profile_data(user_id, updates, source="edicion_usuario", confidence=1.0)
        await update.message.reply_text("Perfil actualizado.")

    async def handle_style_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        style = (" ".join(context.args).strip().lower() if context.args else "")
        if not style:
            current = self.core.get_user_style(user_id)
            await update.message.reply_text(
                f"Estilo actual: {current}. Usa /estilo casual|profesional|breve"
            )
            return

        changed = self.core.set_user_style(user_id, style)
        if changed:
            await update.message.reply_text(f"Estilo actualizado a: {style}")
        else:
            await update.message.reply_text("Estilo invalido. Usa: casual, profesional, breve")

    async def handle_emojis_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        mode = (" ".join(context.args).strip().lower() if context.args else "")
        if not mode:
            prefs = self.core.get_response_preferences(user_id)
            await update.message.reply_text(
                f"Emojis actual: {prefs.get('emojis', 'auto')}. Usa /emojis on|off|auto"
            )
            return

        if self.core.set_response_preference(user_id, "emojis", mode):
            await update.message.reply_text(f"Preferencia de emojis: {mode}")
        else:
            await update.message.reply_text("Valor invalido. Usa /emojis on|off|auto")

    async def handle_chilean_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        mode = (" ".join(context.args).strip().lower() if context.args else "")
        if not mode:
            prefs = self.core.get_response_preferences(user_id)
            await update.message.reply_text(
                f"Modismo chileno actual: {prefs.get('chilean', 'auto')}. Usa /chileno on|off|auto"
            )
            return

        if self.core.set_response_preference(user_id, "chilean", mode):
            await update.message.reply_text(f"Preferencia de modismo chileno: {mode}")
        else:
            await update.message.reply_text("Valor invalido. Usa /chileno on|off|auto")

    async def handle_clear_profile_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        deleted = self.core.clear_profile(user_id)
        if deleted:
            await update.message.reply_text("Perfil eliminado.")
        else:
            await update.message.reply_text("No habia perfil para eliminar.")

    async def handle_clear_memory_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized_private_chat(update):
            return
        user_id = str(update.message.chat_id)
        deleted = self.core.clear_memory(user_id)
        if deleted:
            await update.message.reply_text("Memoria conversacional eliminada.")
        else:
            await update.message.reply_text("No habia memoria para eliminar.")

    async def handle_message(self, update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        if not self.is_authorized_private_chat(update):
            return

        chat = update.effective_chat
        user_id = str(update.message.chat_id)
        user_message = update.message.text.strip()
        preferences = self.core.update_response_preferences(user_id, user_message)
        history = self.core.user_memory.get(user_id, "")

        previous_task = self.core.active_generation_tasks.get(user_id)
        current_task = asyncio.current_task()
        if previous_task and previous_task is not current_task and not previous_task.done():
            previous_task.cancel()

        if current_task is not None:
            self.core.active_generation_tasks[user_id] = current_task

        quick = self.core.fast_response(user_message)
        if quick:
            quick = self.core.finalize_reply(quick, preferences, user_message=user_message, history=history)
            await update.message.reply_text(quick)
            return

        tool_reply = await self.core.try_tool_response(user_message)
        if tool_reply:
            tool_reply = self.core.finalize_reply(tool_reply, preferences, user_message=user_message, history=history)
            await update.message.reply_text(tool_reply)
            self.core.persist_text_interaction(user_id, user_message, tool_reply)
            return

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        profile = self.core.get_profile_for_prompt(user_id)
        style = self.core.get_user_style(user_id)
        relevant_memory = ""
        if len(user_message.split()) > 2:
            relevant_memory = self.core.vector_memory.get_relevant_memories(user_id, user_message)

        prompt = self.core.build_text_prompt(user_message, profile, relevant_memory, history, style, preferences)
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
                        await self._safe_edit_text(msg, final_text[-self.settings.telegram_msg_limit :])
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

            final_text = self.core.sanitize_reply(profile, final_text.strip(), user_message=user_message)
            final_text = await enforce_spanish(
                final_text,
                self.settings.balanced_model,
                self.ollama.generate_text,
                self.logger,
            )
            final_text = self.core.finalize_reply(final_text, preferences, user_message=user_message, history=history)

            should_repair = (
                final_text
                and not self.core.is_reply_on_topic(user_message, final_text)
                and len(user_message.split()) > 4
                and not preferences.get("use_chilean", False)
            )
            if should_repair:
                repair_prompt = self.core.build_precision_prompt(
                    user_message,
                    profile,
                    relevant_memory,
                    history,
                    style,
                    preferences,
                )
                regenerated = await self.ollama.generate_text(repair_prompt, self.settings.balanced_model)
                final_text = self.core.sanitize_reply(profile, regenerated.strip(), user_message=user_message)
                final_text = await enforce_spanish(
                    final_text,
                    self.settings.balanced_model,
                    self.ollama.generate_text,
                    self.logger,
                )
                final_text = self.core.finalize_reply(final_text, preferences, user_message=user_message, history=history)
            if not final_text:
                final_text = "No pude responder."

            parts = self.core.chunk_text(final_text)
            await self._safe_edit_text(msg, parts[0])
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

        if len(user_message.split()) > 2:
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
            self.core.extract_user_info_from_image_caption(user_id, caption)

        except Exception as exc:
            self.logger.exception("Error procesando imagen")
            await status_msg.edit_text(f"No pude analizar la imagen: {exc}")

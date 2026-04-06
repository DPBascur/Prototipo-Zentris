import json

from app.infrastructure.storage import load_json_or_default, save_json_atomic


class AssistantCore:
    def __init__(self, settings, logger, ollama_client, vector_memory):
        self.settings = settings
        self.logger = logger
        self.ollama = ollama_client
        self.vector_memory = vector_memory
        self.user_memory = load_json_or_default(settings.memory_file, {}, logger)
        self.user_profile = load_json_or_default(settings.profile_file, {}, logger)
        self.active_generation_tasks = {}

    def fast_response(self, message: str):
        text = message.lower().strip()
        simple_responses = {
            "hola": "Hola",
            "hi": "Hola",
            "ok": "Ok",
            "gracias": "De nada",
            "vale": "Perfecto",
            "si": "Bien",
            "no": "Entiendo",
        }
        return simple_responses.get(text)

    def choose_model(self, message: str) -> str:
        text = message.lower().strip()
        words = text.split()
        deep_signals = [
            "analiza", "explica", "compara", "estrategia", "arquitectura",
            "debug", "error", "paso a paso", "optimiza", "por que",
        ]
        if len(words) >= 35 or any(signal in text for signal in deep_signals):
            return self.settings.reasoning_model
        if len(words) <= 5:
            return self.settings.fast_model
        return self.settings.balanced_model

    def sanitize_reply(self, profile: dict, reply: str) -> str:
        if profile:
            return reply
        risky_tokens = ["anos", "vivo", "casado", "me gusta", "tu nombre es"]
        lowered = reply.lower()
        if any(token in lowered for token in risky_tokens):
            return "No tengo informacion suficiente sobre ti aun."
        return reply

    def chunk_text(self, text: str):
        limit = self.settings.telegram_msg_limit
        if len(text) <= limit:
            return [text]

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            split_at = remaining.rfind("\n", 0, limit)
            if split_at < int(limit * 0.5):
                split_at = remaining.rfind(" ", 0, limit)
            if split_at < 1:
                split_at = limit

            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        return [chunk for chunk in chunks if chunk]

    def build_text_prompt(self, user_message: str, profile: dict, relevant_memory: str, history: str) -> str:
        profile_text = json.dumps(profile, ensure_ascii=False)
        short_history = history[-400:]
        return f"""
Eres Zentris, un asistente util y directo.

Reglas:
- Responde siempre en espanol.
- No inventes datos del usuario.
- Si falta informacion personal, dilo claramente.

Perfil:
{profile_text if profile else "VACIO"}

Memoria relevante:
{relevant_memory if relevant_memory else "NINGUNA"}

Conversacion reciente:
{short_history if short_history else "NINGUNA"}

Usuario: {user_message}
Zentris:
"""

    def build_vision_prompt(self, caption: str) -> str:
        if caption:
            return (
                "Responde siempre en espanol. "
                "Analiza la imagen siguiendo la instruccion del usuario. "
                "Responde de forma directa y concreta a lo que se pregunta. "
                f"Instruccion del usuario: {caption}"
            )

        return (
            "Responde siempre en espanol. Analiza esta imagen en espanol. "
            "Describe lo importante, objetos, contexto y posibles riesgos si los hay. "
            "Si hay texto visible, extraelo y resumelo."
        )

    def persist_text_interaction(self, user_id: str, user_message: str, final_text: str) -> None:
        history = self.user_memory.get(user_id, "")
        new_history = history + f"\nUsuario: {user_message}\nZentris: {final_text}"
        self.user_memory[user_id] = new_history[-self.settings.max_history:]
        save_json_atomic(self.settings.memory_file, self.user_memory, self.logger)

    def persist_image_interaction(self, user_id: str, caption: str, reply: str) -> None:
        history = self.user_memory.get(user_id, "")
        new_history = history + f"\nUsuario: [Imagen] {caption if caption else '(sin caption)'}\nZentris: {reply}"
        self.user_memory[user_id] = new_history[-self.settings.max_history:]
        save_json_atomic(self.settings.memory_file, self.user_memory, self.logger)

    async def extract_user_info_async(self, user_id: str, message: str) -> None:
        if len(message.split()) < 6:
            return

        prompt = f"""
Extrae informacion del usuario en JSON.

Mensaje:
\"{message}\"

Reglas:
- Solo JSON valido
- No inventar
- Si no hay info: {{}}
"""

        try:
            extracted = await self.ollama.generate_json(prompt, self.settings.fast_model)
            if not extracted:
                return

            if user_id not in self.user_profile:
                self.user_profile[user_id] = {}
            self.user_profile[user_id].update(extracted)
            save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        except Exception as exc:
            self.logger.warning("No se pudo extraer perfil para %s: %s", user_id, exc)

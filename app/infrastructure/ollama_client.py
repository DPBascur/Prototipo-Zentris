import json
import re

import httpx


class OllamaClient:
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        self.available_models: set[str] = set()

    def validate_models(self) -> None:
        try:
            with httpx.Client(timeout=10) as client:
                tags_response = client.get(self.settings.ollama_tags_url)
                tags_response.raise_for_status()
                payload = tags_response.json()
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo conectar a Ollama en {self.settings.ollama_tags_url}: {exc}"
            )

        available = set()
        for model in payload.get("models", []):
            name = model.get("name", "")
            if name:
                available.add(name)
                available.add(name.split(":", 1)[0])

        self.available_models = available

        required = {
            self.settings.fast_model,
            self.settings.balanced_model,
            self.settings.reasoning_model,
        }
        missing = [name for name in required if name not in available]
        if missing:
            raise RuntimeError(f"Modelos no disponibles en Ollama: {', '.join(missing)}")

        if self.settings.vision_model not in available:
            self.logger.warning(
                "VISION_MODEL '%s' no esta disponible en Ollama. El analisis de imagenes devolvera aviso.",
                self.settings.vision_model,
            )

    def _build_options(self, temperature: float | None = None) -> dict:
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if self.settings.force_gpu:
            # Ollama puede ignorar esta opcion segun el backend, pero ayuda a evitar fallback a CPU.
            options["num_gpu"] = 1
        return options

    async def stream_text(self, prompt: str, model: str, temperature: float = 0.25):
        timeout = httpx.Timeout(self.settings.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self.settings.ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": self._build_options(temperature),
                },
            ) as response:
                response.raise_for_status()

                full_text = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            full_text += chunk
                            yield full_text
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def generate_json(self, prompt: str, model: str) -> dict:
        timeout = httpx.Timeout(self.settings.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.settings.ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": self._build_options(),
                },
            )
            response.raise_for_status()

        payload = response.json()
        text = payload.get("response", "{}").strip()
        parsed = self._parse_json_object(text)
        return parsed if isinstance(parsed, dict) else {}

    def _parse_json_object(self, text: str) -> dict:
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first : last + 1]
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        return {}

    async def generate_text(self, prompt: str, model: str) -> str:
        timeout = httpx.Timeout(self.settings.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.settings.ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": self._build_options(0.2),
                },
            )
            response.raise_for_status()

        payload = response.json()
        return payload.get("response", "").strip()

    async def generate_vision_text(self, prompt: str, image_b64: str, model: str) -> str:
        timeout = httpx.Timeout(self.settings.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.settings.ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": self._build_options(0.2),
                },
            )
            if response.status_code >= 400:
                detail = ""
                try:
                    detail = response.json().get("error", "")
                except Exception:
                    detail = response.text
                raise RuntimeError(f"{response.status_code} {detail}".strip())

        payload = response.json()
        return payload.get("response", "").strip()

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import httpx

from app.infrastructure.storage import load_json_or_default, save_json_atomic


class AssistantCore:
    ALLOWED_PROFILE_FIELDS = {
        "nombre",
        "edad",
        "city",
        "nationality",
        "idioma_preferido",
        "objetivo",
        "rol_creador",
        "studies",
        "intereses",
        "gato",
    }
    ALLOWED_STYLES = {"casual", "profesional", "breve"}
    ALLOWED_PREF_VALUES = {"on", "off", "auto"}

    def _emoji_pattern(self):
        return re.compile(
            "["
            "\U0001F300-\U0001F5FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "\U00002700-\U000027BF"
            "\U00002600-\U000026FF"
            "]+",
            flags=re.UNICODE,
        )

    def __init__(self, settings, logger, ollama_client, vector_memory):
        self.settings = settings
        self.logger = logger
        self.ollama = ollama_client
        self.vector_memory = vector_memory
        self.chilean_dictionary = self._load_chilean_dictionary(settings.base_dir)
        self.user_memory = load_json_or_default(settings.memory_file, {}, logger)
        raw_profile = load_json_or_default(settings.profile_file, {}, logger)
        self.user_profile = self._sanitize_profile_store(raw_profile)
        if self.user_profile != raw_profile:
            save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        self.active_generation_tasks = {}

    def _default_chilean_dictionary(self) -> dict:
        return {
            "nicknames": ["hermano", "compa", "bro", "socio", "compare"],
            "direct_replacements": {
                "amigo": "{nick}",
                "Amigo": "{nick_cap}",
                "amiga": "{nick}",
                "Amiga": "{nick_cap}",
                "te ayudo": "te apano",
                "te ayudo al tiro": "te apano al tiro",
                "que necesitas": "que necesitai",
                "¿qué necesitas?": "que necesitai?",
                "¿que necesitas?": "que necesitai?",
                "todo bien": "todo piola",
                "de acuerdo": "dale",
                "entiendo": "cacho",
                "perfecto": "bacan",
                "estoy aqui para ayudarte": "aqui estoy pa apanar",
            },
            "context_templates": {
                "thanks": [
                    "De una, {nick}. Cualquier cosa me dices.",
                    "Dale {nick}, pa eso estamos.",
                    "Bacan, {nick}. Si queris seguimos.",
                ],
                "greeting": [
                    "Wena {nick}, todo piola por aca. Y tu?",
                    "Wena {nick}, aqui ando al tiro.",
                    "Que tal {nick}, te leo.",
                ],
            },
        }

    def _load_chilean_dictionary(self, base_dir: Path) -> dict:
        default = self._default_chilean_dictionary()
        file_path = base_dir / "app" / "data" / "chilean_modismos.json"
        try:
            if not file_path.exists():
                return default
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default

            nicknames = raw.get("nicknames", default["nicknames"])
            if not isinstance(nicknames, list) or not nicknames:
                nicknames = default["nicknames"]

            direct_replacements = raw.get("direct_replacements", default["direct_replacements"])
            if not isinstance(direct_replacements, dict):
                direct_replacements = default["direct_replacements"]

            context_templates = raw.get("context_templates", default["context_templates"])
            if not isinstance(context_templates, dict):
                context_templates = default["context_templates"]

            return {
                "nicknames": [str(x) for x in nicknames if isinstance(x, str) and x.strip()],
                "direct_replacements": {
                    str(k): str(v)
                    for k, v in direct_replacements.items()
                    if isinstance(k, str) and isinstance(v, str)
                },
                "context_templates": {
                    str(k): [str(item) for item in v if isinstance(item, str)]
                    for k, v in context_templates.items()
                    if isinstance(k, str) and isinstance(v, list)
                },
            }
        except Exception as exc:
            self.logger.warning("No se pudo cargar diccionario chileno, usando fallback: %s", exc)
            return default

    def _now_iso(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _clean_text(self, value, max_len: int = 80) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        return text[:max_len]

    def _clean_list_of_text(self, value, max_items: int = 8, max_len: int = 60) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        seen = set()
        for item in value:
            text = self._clean_text(item, max_len=max_len)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _sanitize_profile_payload(self, payload: dict) -> dict:
        merged = {}
        if not isinstance(payload, dict):
            return merged

        nombre = self._clean_text(payload.get("nombre"), max_len=40)
        if nombre:
            merged["nombre"] = nombre

        edad = payload.get("edad")
        if isinstance(edad, int) and 5 <= edad <= 110:
            merged["edad"] = edad

        city = self._clean_text(payload.get("city"), max_len=80)
        if city:
            merged["city"] = city

        nationality = self._clean_text(payload.get("nationality"), max_len=40)
        if nationality:
            merged["nationality"] = nationality

        idioma_preferido = self._clean_text(payload.get("idioma_preferido"), max_len=20)
        if idioma_preferido:
            merged["idioma_preferido"] = idioma_preferido

        objetivo = self._clean_text(payload.get("objetivo"), max_len=120)
        if objetivo:
            merged["objetivo"] = objetivo

        rol_creador = self._clean_text(payload.get("rol_creador"), max_len=60)
        if rol_creador:
            merged["rol_creador"] = rol_creador

        studies = self._clean_list_of_text(payload.get("studies"), max_items=10, max_len=60)
        if studies:
            merged["studies"] = studies

        intereses = self._clean_list_of_text(payload.get("intereses"), max_items=12, max_len=50)
        if intereses:
            merged["intereses"] = intereses

        pet = payload.get("gato")
        if isinstance(pet, dict):
            pet_name = self._clean_text(pet.get("nombre"), max_len=40)
            if pet_name:
                merged["gato"] = {"nombre": pet_name}

        return merged

    def _sanitize_meta(self, meta: dict, allowed_fields: set[str]) -> dict:
        clean_meta = {}
        if not isinstance(meta, dict):
            return clean_meta

        for field, info in meta.items():
            if field not in allowed_fields or not isinstance(info, dict):
                continue
            confidence = info.get("confidence", 0.7)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.7
            confidence = max(0.0, min(1.0, confidence))
            source = self._clean_text(info.get("source"), max_len=40) or "legacy"
            updated_at = self._clean_text(info.get("updated_at"), max_len=40) or self._now_iso()
            clean_meta[field] = {
                "confidence": confidence,
                "source": source,
                "updated_at": updated_at,
            }
        return clean_meta

    def _sanitize_profile_store(self, profile_store: dict) -> dict:
        if not isinstance(profile_store, dict):
            return {}

        clean_store = {}
        for user_id, payload in profile_store.items():
            if not isinstance(user_id, str):
                continue
            if not isinstance(payload, dict):
                continue

            if "data" in payload:
                data_raw = payload.get("data", {})
                style_raw = payload.get("style", "casual")
                meta_raw = payload.get("meta", {})
                prefs_raw = payload.get("preferences", {})
            else:
                data_raw = payload
                style_raw = payload.get("style", "casual")
                meta_raw = {
                    key: {"confidence": 0.7, "source": "legacy", "updated_at": self._now_iso()}
                    for key in data_raw.keys()
                    if key in self.ALLOWED_PROFILE_FIELDS
                }
                prefs_raw = payload.get("preferences", {})

            clean_data = self._sanitize_profile_payload(data_raw)
            if not clean_data:
                continue

            style = self._clean_text(style_raw, max_len=20) or "casual"
            style = style if style in self.ALLOWED_STYLES else "casual"
            meta = self._sanitize_meta(meta_raw, set(clean_data.keys()))

            for key in clean_data.keys():
                if key not in meta:
                    meta[key] = {
                        "confidence": 0.7,
                        "source": "legacy",
                        "updated_at": self._now_iso(),
                    }

            preferences = {
                "emojis": "auto",
                "chilean": "auto",
            }
            if isinstance(prefs_raw, dict):
                emojis_pref = self._clean_text(prefs_raw.get("emojis"), max_len=10)
                chilean_pref = self._clean_text(prefs_raw.get("chilean"), max_len=10)
                if emojis_pref in self.ALLOWED_PREF_VALUES:
                    preferences["emojis"] = emojis_pref
                if chilean_pref in self.ALLOWED_PREF_VALUES:
                    preferences["chilean"] = chilean_pref

            clean_store[user_id] = {
                "data": clean_data,
                "meta": meta,
                "style": style,
                "preferences": preferences,
            }

        return clean_store

    def _get_profile_entry(self, user_id: str) -> dict:
        entry = self.user_profile.get(user_id)
        if not isinstance(entry, dict):
            entry = {"data": {}, "meta": {}, "style": "casual"}
            self.user_profile[user_id] = entry
        entry.setdefault("data", {})
        entry.setdefault("meta", {})
        entry.setdefault("style", "casual")
        entry.setdefault("preferences", {"emojis": "auto", "chilean": "auto"})
        if entry["style"] not in self.ALLOWED_STYLES:
            entry["style"] = "casual"
        if not isinstance(entry["preferences"], dict):
            entry["preferences"] = {"emojis": "auto", "chilean": "auto"}
        if entry["preferences"].get("emojis") not in self.ALLOWED_PREF_VALUES:
            entry["preferences"]["emojis"] = "auto"
        if entry["preferences"].get("chilean") not in self.ALLOWED_PREF_VALUES:
            entry["preferences"]["chilean"] = "auto"
        return entry

    def _contains_any(self, text: str, terms: list[str]) -> bool:
        return any(term in text for term in terms)

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        normalized = unicodedata.normalize("NFKD", text)
        without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", without_accents)

    def _looks_like_datetime_question(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        patterns = [
            r"\bque\s+hora\s+es\b",
            r"\bcual\s+es\s+la\s+hora\b",
            r"\bhora\s+actual\b",
            r"\bque\s+fecha\s+es\b",
            r"\bcual\s+es\s+la\s+fecha\b",
            r"\bfecha\s+actual\b",
            r"\bque\s+dia\s+es\s+hoy\b",
            r"\bhoy\s+es\s+que\s+dia\b",
            r"^hora\??$",
            r"^fecha\??$",
        ]
        return any(re.search(pat, normalized) for pat in patterns)

    def _wants_weather(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        weather_patterns = [
            r"\bclima\b.*\b(en|de)\b",
            r"\btemperatura\b.*\b(en|de)\b",
            r"\bque\s+tiempo\s+hace\b.*\b(en|de)\b",
            r"\bcomo\s+esta\s+el\s+clima\b.*\b(en|de)\b",
        ]
        return any(re.search(pat, normalized) for pat in weather_patterns)

    def _wants_improvement_ideas(self, user_message: str) -> bool:
        text = self._normalize_text(user_message)
        patterns = [
            r"\bmejorar\b",
            r"\bsubir\s+de\s+nivel\b",
            r"\bsiguiente\s+nivel\b",
            r"\boptimizar\b",
            r"\brecomendaciones\b",
            r"\bideas\b",
            r"\bcomo\s+lo\s+mejoramos\b",
            r"\bque\s+podemos\s+mejorar\b",
        ]
        return any(re.search(pat, text) for pat in patterns)

    def _is_positive_feedback(self, user_message: str) -> bool:
        text = self._normalize_text(user_message)
        patterns = [
            r"\bgracias\b",
            r"\bfeliz\b",
            r"\bcontento\b",
            r"\bbuen\s+trabajo\b",
            r"\bfunciona\s+mejor\b",
            r"\brespondes\s+rapido\b",
            r"\bme\s+encanta\b",
        ]
        return any(re.search(pat, text) for pat in patterns)

    def update_response_preferences(self, user_id: str, user_message: str) -> dict:
        entry = self._get_profile_entry(user_id)
        prefs = entry["preferences"]
        text = user_message.lower()
        changed = False

        if self._contains_any(text, ["sin emojis", "no emojis", "sin emoji", "sin emoticon"]):
            prefs["emojis"] = "off"
            changed = True
        elif self._contains_any(text, ["con emojis", "usa emojis", "pon emojis", "quiero emojis"]):
            prefs["emojis"] = "on"
            changed = True

        if self._contains_any(text, ["sin modismos", "espanol neutro", "español neutro", "sin chileno"]):
            prefs["chilean"] = "off"
            changed = True
        elif self._contains_any(text, ["modismo chileno", "habla chileno", "chileno", "chilenismos"]):
            prefs["chilean"] = "on"
            changed = True

        if changed:
            save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)

        emojis_mode = prefs.get("emojis", "auto")
        chilean_mode = prefs.get("chilean", "auto")

        use_emojis = emojis_mode == "on" or (emojis_mode == "auto" and self.get_user_style(user_id) == "casual")

        profile_data = entry.get("data", {}) if isinstance(entry.get("data"), dict) else {}
        is_chilean_user = False
        nationality = str(profile_data.get("nationality", "")).lower()
        city = str(profile_data.get("city", "")).lower()
        if "chil" in nationality or "chile" in city or "temuco" in city:
            is_chilean_user = True
        use_chilean = chilean_mode == "on" or (chilean_mode == "auto" and is_chilean_user)

        return {"use_emojis": use_emojis, "use_chilean": use_chilean}

    def get_response_preferences(self, user_id: str) -> dict:
        entry = self._get_profile_entry(user_id)
        prefs = entry.get("preferences", {})
        return {
            "emojis": prefs.get("emojis", "auto"),
            "chilean": prefs.get("chilean", "auto"),
        }

    def set_response_preference(self, user_id: str, key: str, value: str) -> bool:
        key = (key or "").strip().lower()
        value = (value or "").strip().lower()
        if key not in {"emojis", "chilean"}:
            return False
        if value not in self.ALLOWED_PREF_VALUES:
            return False

        entry = self._get_profile_entry(user_id)
        entry["preferences"][key] = value
        save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        return True

    def _strip_emojis(self, text: str) -> str:
        return self._emoji_pattern().sub("", text).strip()

    def _has_emoji(self, text: str) -> bool:
        if not text:
            return False
        return bool(self._emoji_pattern().search(text))

    def _last_assistant_reply(self, history: str) -> str:
        if not history:
            return ""
        matches = re.findall(r"Zentris:\s*(.+)", history)
        return matches[-1].strip() if matches else ""

    def _is_greeting_message(self, user_message: str) -> bool:
        text = (user_message or "").lower().strip()
        greeting_terms = [
            "hola",
            "wena",
            "buenas",
            "que tal",
            "qué tal",
            "como estai",
            "como estas",
            "cómo estás",
            "hermano",
        ]
        return len(text.split()) <= 7 and any(term in text for term in greeting_terms)

    def _remove_repeated_greeting(self, reply: str, user_message: str, history: str) -> str:
        if not reply:
            return reply
        if not history:
            return reply
        if self._is_greeting_message(user_message):
            return reply

        cleaned = re.sub(
            r"^(hola(?:\s+amigo|\s+hermano)?|wena(?:\s+hermano)?|buenas(?:\s+dias|\s+tardes)?|que\s+tal|qué\s+tal)[,\.!\s]*",
            "",
            reply,
            flags=re.IGNORECASE,
        ).strip()

        msg_lower = (user_message or "").lower()
        if "como te llamas" not in msg_lower and "cómo te llamas" not in msg_lower:
            cleaned = re.sub(r"^soy\s+zentris[,\s]*", "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned if cleaned else reply

    def _pick_chilean_nick(self, user_message: str, last_reply: str) -> str:
        user_lower = (user_message or "").lower()
        if "hermano" in user_lower:
            return "hermano"
        if "weon" in user_lower or "weón" in user_lower:
            return "weon"

        nicknames = self.chilean_dictionary.get("nicknames", [])
        if not nicknames:
            nicknames = ["hermano", "compa", "bro", "socio"]

        for nick in nicknames:
            if nick in (last_reply or "").lower():
                idx = nicknames.index(nick)
                return nicknames[(idx + 1) % len(nicknames)]

        return nicknames[0]

    def _detect_chilean_intent(self, user_message: str) -> str:
        text = (user_message or "").lower().strip()
        if any(w in text for w in ["gracias", "vale", "bacan", "bkn"]):
            return "thanks"
        if any(w in text for w in ["hola", "wena", "que tal", "qué tal", "como estai", "como estas", "cómo estás"]):
            return "greeting"
        return "general"

    def _pick_template(self, templates: list[str], last_reply: str) -> str | None:
        if not templates:
            return None
        if not last_reply:
            return templates[0]
        last_lower = last_reply.lower()
        for item in templates:
            if item.lower() not in last_lower:
                return item
        return templates[0]

    def _append_levelup_hint(self, reply: str) -> str:
        if not reply:
            return reply
        lowered = reply.lower()
        if "si quieres" in lowered and "mejora" in lowered:
            return reply
        hint = "\n\nSi quieres, te propongo 2 mejoras concretas para subir aun mas el nivel."
        return f"{reply.rstrip()}{hint}"

    def _pick_context_emoji(self, user_message: str, reply: str) -> str | None:
        text = self._normalize_text(f"{user_message} {reply}")
        if any(re.search(pat, text) for pat in [r"\berror\b", r"\bfallo\b", r"\bno\s+funciona\b", r"\bproblema\b"]):
            return "🛠️"
        if any(re.search(pat, text) for pat in [r"\bidea\b", r"\bmejora\b", r"\brecomend", r"\btip\b"]):
            return "💡"
        if any(re.search(pat, text) for pat in [r"\bgracias\b", r"\bperfecto\b", r"\bexcelente\b", r"\bbacan\b"]):
            return "🙌"
        if self._is_greeting_message(user_message):
            return "🙂"
        return None

    def _apply_chilean_modismos(self, reply: str, user_message: str, history: str) -> str:
        if not reply:
            return reply

        last_reply = self._last_assistant_reply(history)
        nick = self._pick_chilean_nick(user_message, last_reply)

        replacements = self.chilean_dictionary.get("direct_replacements", {})
        if not isinstance(replacements, dict):
            replacements = {}

        out = reply
        for src, dst in replacements.items():
            mapped = (
                dst.replace("{nick}", nick)
                .replace("{nick_cap}", nick.capitalize())
            )
            out = out.replace(src, mapped)

        context_templates = self.chilean_dictionary.get("context_templates", {})
        intent = self._detect_chilean_intent(user_message)
        templates = context_templates.get(intent, []) if isinstance(context_templates, dict) else []

        if templates and len((user_message or "").split()) <= 7:
            chosen = self._pick_template(templates, last_reply)
            if chosen:
                templated = (
                    chosen.replace("{nick}", nick)
                    .replace("{nick_cap}", nick.capitalize())
                )
                out = templated

        # Si ya saludo, evita repetirlo en exceso en modo chileno.
        out = re.sub(r"^(hola\s+\w+[,\s]*)+", "", out, flags=re.IGNORECASE).strip()
        if not out:
            out = f"Dale {nick}, te leo." if nick != "weon" else "Dale weon, te leo."

        return out

    def finalize_reply(
        self,
        reply: str,
        preferences: dict | None = None,
        user_message: str = "",
        history: str = "",
    ) -> str:
        if not reply:
            return reply

        reply = self._strip_internal_leaks(reply)

        reply = self._remove_repeated_greeting(reply, user_message, history)

        if preferences and preferences.get("use_chilean", False):
            replacements = {
                "Estoy funcionando correctamente": "Todo piola por aca",
                "estoy funcionando correctamente": "todo piola por aca",
                "listo para ayudarte": "al tiro pa ayudarte",
                "¿Qué puedo hacer por ti hoy?": "Que necesitas? te ayudo al tiro.",
                "¿Qué puedo hacer por ti?": "Que necesitas?",
                "Sí, entiendo que": "Dale, cacho que",
                "Si, entiendo que": "Dale, cacho que",
            }
            for src, dst in replacements.items():
                reply = reply.replace(src, dst)
            reply = self._apply_chilean_modismos(reply, user_message, history)

        if self._wants_improvement_ideas(user_message) or self._is_positive_feedback(user_message):
            reply = self._append_levelup_hint(reply)

        if preferences and not preferences.get("use_emojis", True):
            return self._strip_emojis(reply)

        if preferences and preferences.get("use_emojis", True):
            last_reply = self._last_assistant_reply(history)
            if self._has_emoji(last_reply):
                reply = self._strip_emojis(reply)
            else:
                emojis = self._emoji_pattern().findall(reply)
                if len(emojis) > 1:
                    base = self._strip_emojis(reply)
                    reply = f"{base} {emojis[0]}".strip()
                elif len(emojis) == 0:
                    context_emoji = self._pick_context_emoji(user_message, reply)
                    if context_emoji:
                        reply = f"{reply.strip()} {context_emoji}"

        return reply

    def get_user_style(self, user_id: str) -> str:
        entry = self.user_profile.get(user_id, {})
        style = entry.get("style") if isinstance(entry, dict) else None
        return style if style in self.ALLOWED_STYLES else "casual"

    def set_user_style(self, user_id: str, style: str) -> bool:
        style = (style or "").strip().lower()
        if style not in self.ALLOWED_STYLES:
            return False
        entry = self._get_profile_entry(user_id)
        entry["style"] = style
        save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        return True

    def get_profile_data(self, user_id: str, min_confidence: float = 0.0) -> dict:
        entry = self.user_profile.get(user_id)
        if not isinstance(entry, dict):
            return {}
        data = entry.get("data", {})
        meta = entry.get("meta", {})
        if not isinstance(data, dict):
            return {}
        if min_confidence <= 0:
            return dict(data)

        filtered = {}
        for key, value in data.items():
            info = meta.get(key, {}) if isinstance(meta, dict) else {}
            confidence = info.get("confidence", 0.0) if isinstance(info, dict) else 0.0
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            if confidence >= min_confidence:
                filtered[key] = value
        return filtered

    def get_profile_for_prompt(self, user_id: str) -> dict:
        return self.get_profile_data(user_id, min_confidence=0.7)

    def upsert_profile_data(self, user_id: str, payload: dict, source: str, confidence: float) -> bool:
        clean_payload = self._sanitize_profile_payload(payload)
        if not clean_payload:
            return False

        entry = self._get_profile_entry(user_id)
        now = self._now_iso()
        source_clean = self._clean_text(source, max_len=40) or "unknown"
        conf = max(0.0, min(1.0, float(confidence)))

        for key, value in clean_payload.items():
            entry["data"][key] = value
            entry["meta"][key] = {
                "confidence": conf,
                "source": source_clean,
                "updated_at": now,
            }

        save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        return True

    def _field_evidence_terms(self, field: str) -> list[str]:
        evidence = {
            "nombre": ["me llamo", "mi nombre", "soy"],
            "edad": ["tengo", "anos", "años", "edad"],
            "city": ["vivo en", "soy de", "ciudad", "temuco", "santiago"],
            "nationality": ["chileno", "chilena", "nacionalidad", "soy de"],
            "idioma_preferido": ["idioma", "prefiero", "hablo"],
            "objetivo": ["objetivo", "quiero", "meta", "busco"],
            "rol_creador": ["soy", "rol", "trabajo como", "me dedico"],
            "studies": ["estudio", "universidad", "carrera", "ingenier"],
            "intereses": ["me gusta", "interesa", "hobby", "interes"],
            "gato": ["mi gato", "mi gatito", "mi mascota", "se llama"],
        }
        return evidence.get(field, [])

    def _is_low_quality_profile_value(self, value) -> bool:
        if isinstance(value, dict):
            value = " ".join(str(v) for v in value.values())
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        text = self._normalize_text(str(value))
        if not text or len(text) < 3:
            return True
        bad_chunks = [
            "no hermano",
            "no se",
            "no tengo",
            "nada",
            "error",
            "pregunta anterior",
            "codigo",
            "csharp",
        ]
        return any(chunk in text for chunk in bad_chunks)

    def _filter_extracted_profile(self, extracted: dict, message: str) -> dict:
        if not isinstance(extracted, dict):
            return {}

        msg = self._normalize_text(message)
        filtered = {}
        for field, value in extracted.items():
            if field not in self.ALLOWED_PROFILE_FIELDS:
                continue
            if self._is_low_quality_profile_value(value):
                continue
            terms = self._field_evidence_terms(field)
            has_evidence = any(term in msg for term in terms)
            if not has_evidence and field not in {"nombre"}:
                continue
            filtered[field] = value
        return filtered

    def parse_profile_edit_text(self, raw_text: str) -> dict:
        updates = {}
        if not raw_text:
            return updates

        segments = [seg.strip() for seg in raw_text.split(";") if seg.strip()]
        for seg in segments:
            if "=" not in seg:
                continue
            key, value = seg.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key not in self.ALLOWED_PROFILE_FIELDS:
                continue
            if key in {"studies", "intereses"}:
                updates[key] = [item.strip() for item in value.split(",") if item.strip()]
            elif key == "edad":
                try:
                    updates[key] = int(value)
                except Exception:
                    continue
            elif key == "gato":
                updates[key] = {"nombre": value}
            else:
                updates[key] = value

        return updates

    def fast_response(self, message: str):
        text = message.lower().strip()
        simple_responses = {
            "hola": "Hola, aqui estoy",
            "hola zentris": "Aqui estoy, te escucho",
            "zentris": "Te escucho",
            "wena": "Wena, aqui ando",
            "wena hermano": "Wena hermano, aqui ando al tiro",
            "wena hermano como estai": "Wena hermano, todo piola por aca. Y tu?",
            "wena hermano como estai?": "Wena hermano, todo piola por aca. Y tu?",
            "como estai": "Todo piola por aca, y tu?",
            "como estai?": "Todo piola por aca, y tu?",
            "que tal": "Todo bien por aca, cuentame",
            "qué tal": "Todo bien por aca, cuentame",
            "hi": "Hola",
            "ok": "Ok",
            "gracias": "De nada",
            "vale": "Perfecto",
            "si": "Bien",
            "no": "Entiendo",
            "como estas": "Todo bien por aqui, listo para ayudarte",
            "como estas?": "Todo bien por aqui, listo para ayudarte",
            "como te llamas": "Soy Zentris",
            "como te llamas?": "Soy Zentris",
        }
        return simple_responses.get(text)

    def choose_model(self, message: str) -> str:
        text = message.lower().strip()
        words = text.split()
        social_signals = ["hola", "como estas", "que tal", "gracias", "como te llamas", "buenas"]
        math_signals = [
            "ecuacion",
            "sistema",
            "matriz",
            "escalona",
            "despeja",
            "resolver",
            "resuelve",
            "deriva",
            "integral",
            "sumar",
            "restar",
            "multiplicar",
            "dividir",
            "x",
            "y",
            "z",
        ]
        deep_signals = [
            "analiza",
            "explica",
            "compara",
            "estrategia",
            "arquitectura",
            "debug",
            "error",
            "paso a paso",
            "optimiza",
            "por que",
        ]
        if any(signal in text for signal in social_signals):
            return self.settings.balanced_model
        if any(signal in text for signal in math_signals):
            return self.settings.reasoning_model
        if len(words) >= 35 or any(signal in text for signal in deep_signals):
            return self.settings.reasoning_model
        if len(words) <= 5:
            return self.settings.fast_model
        return self.settings.balanced_model

    def _style_instruction(self, style: str) -> str:
        if style == "profesional":
            return "Tono profesional, claro y estructurado."
        if style == "breve":
            return "Tono directo, respuestas muy cortas y concretas."
        return "Tono cercano, natural y relajado."

    def _strip_internal_leaks(self, reply: str) -> str:
        if not reply:
            return reply

        text = reply
        leak_markers = [
            r"\n\s*Usuario:\s*",
            r"\n\s*Zentris:\s*",
            r"\n\s*Perfil\s*\(",
            r"\n\s*Memoria\s+relevante:\s*",
            r"\n\s*Conversacion\s+reciente:\s*",
            r"\n\s*Reglas:\s*",
        ]

        cut_positions = []
        for marker in leak_markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                cut_positions.append(match.start())

        if cut_positions:
            text = text[: min(cut_positions)].strip()

        if not text:
            return "Perdon, se mezclo contexto. Te respondo de nuevo de forma directa."

        return text

    def _has_internal_leak_markers(self, reply: str) -> bool:
        if not reply:
            return False
        lowered = reply.lower()
        markers = [
            "usuario:",
            "zentris:",
            "memoria relevante:",
            "conversacion reciente:",
            "perfil (solo alta confianza):",
        ]
        return any(marker in lowered for marker in markers)

    def sanitize_reply(self, profile: dict, reply: str, user_message: str = "") -> str:
        reply = self._strip_internal_leaks(reply)
        msg = user_message.lower().strip()
        if "que sabes de mi" in msg or "que sabes de mí" in msg:
            return "Se de ti solo lo que me compartes en esta conversacion."

        risky_tokens = ["anos", "vivo", "casado", "me gusta", "tu nombre es"]
        lowered = reply.lower()

        rigid_patterns = [
            "como asistente virtual",
            "no dispongo de informacion personal",
            "consentimiento explicito",
            "reglas establecidas",
            "dentro del alcance permitido",
            "no tengo emociones ni sentimientos",
        ]
        if any(pat in lowered for pat in rigid_patterns):
            if any(x in msg for x in ["como estas", "como te llamas", "hola", "gracias"]):
                quick = self.fast_response(msg)
                if quick:
                    return quick
            reply = re.sub(r"(?i)como asistente virtual[,\s]*", "", reply).strip()
            reply = re.sub(r"(?i)sin necesidad de almacenar informacion personal\.?", "", reply).strip()
            reply = re.sub(r"\s{2,}", " ", reply).strip()
            lowered = reply.lower()

        is_math_prompt = any(token in msg for token in ["ecuacion", "sistema", "matriz", "escalona", "resuelve", "despeja", " x", " y"])
        if is_math_prompt:
            has_math_in_reply = any(token in lowered for token in ["x", "y", "=", "ecuacion", "sistema", "matriz", "solucion", "resultado"])
            if not has_math_in_reply:
                return "Creo que me desvie. Reenvia la ecuacion completa y la resuelvo paso a paso."

        if profile:
            interests = profile.get("intereses", []) if isinstance(profile, dict) else []
            interests_text = " ".join(str(item).lower() for item in interests)
            asked_about_center = any(token in user_message.lower() for token in ["centro", "centrista", "centroismo"])
            if asked_about_center and ("centro" in lowered or "centrista" in lowered or "centroismo" in lowered):
                if "centro" not in interests_text:
                    return "No tengo datos en tu perfil para afirmar eso."
            return reply

        if any(token in lowered for token in risky_tokens):
            return "No tengo informacion suficiente sobre ti aun."
        return reply

    def is_reply_on_topic(self, user_message: str, reply: str) -> bool:
        if self._has_internal_leak_markers(reply):
            return False

        user_tokens = re.findall(r"[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]{3,}", user_message.lower())
        if not user_tokens:
            return True
        reply_text = reply.lower()
        overlap = sum(1 for token in set(user_tokens) if token in reply_text)

        if any(k in user_message.lower() for k in ["ecuacion", "sistema", "resuelve", "despeja"]):
            return overlap >= 1 and any(k in reply_text for k in ["x", "y", "=", "soluci", "resultado"])

        return overlap >= 1

    def build_precision_prompt(
        self,
        user_message: str,
        profile: dict,
        relevant_memory: str,
        history: str,
        style: str,
        preferences: dict | None = None,
    ) -> str:
        base = self.build_text_prompt(user_message, profile, relevant_memory, history, style, preferences)
        return (
            base
            + "\n\nCorreccion obligatoria: responde SOLO a la pregunta actual."
            + " No agregues temas no solicitados ni disclaimers."
            + " Conserva el tono indicado (incluyendo modismo chileno y emojis si corresponde)."
        )

    async def try_tool_response(self, user_message: str) -> str | None:
        text = user_message.strip()
        normalized = self._normalize_text(text)

        music = await self._tool_music(user_message)
        if music:
            return music

        if self._looks_like_datetime_question(normalized):
            now = datetime.now()
            return f"Hoy es {now.strftime('%d/%m/%Y')} y son las {now.strftime('%H:%M')}."

        if self._wants_weather(normalized):
            weather = await self._tool_weather(user_message)
            if weather:
                return weather

        if any(k in normalized for k in ["ecuacion", "resuelve", "sistema", "despeja", "simplifica", "deriva", "integral", "="]):
            math_answer = self._tool_math(user_message)
            if math_answer:
                return math_answer

        return None

    async def _tool_music(self, user_message: str) -> str | None:
        text = user_message.strip()
        lowered = text.lower()
        music_signals = ["cancion", "canciones", "tema", "temas", "disco", "album"]
        if not any(sig in lowered for sig in music_signals):
            return None

        artist = None
        patterns = [
            r"(?:canciones|temas|cancion|tema)\s+(?:del|de la|de)\s+(?:cantante\s+)?([\w\s\-áéíóúñÁÉÍÓÚÑ\.]{2,60})",
            r"(?:de|del)\s+(?:cantante\s+)?([\w\s\-áéíóúñÁÉÍÓÚÑ\.]{2,60})\s+(?:canciones|temas)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                artist = m.group(1).strip(" .")
                break

        if not artist:
            m = re.search(r"cantante\s+([\w\s\-áéíóúñÁÉÍÓÚÑ\.]{2,60})", text, re.IGNORECASE)
            if m:
                artist = m.group(1).strip(" .")

        if not artist:
            return None

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    "https://itunes.apple.com/search",
                    params={"term": artist, "entity": "song", "limit": 15},
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            self.logger.warning("Fallo herramienta musica: %s", exc)
            return None

        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            return f"No encontre canciones verificables para '{artist}'."

        artist_lower = artist.lower()
        songs = []
        seen = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            track = self._clean_text(item.get("trackName"), max_len=80)
            artist_name = self._clean_text(item.get("artistName"), max_len=80)
            if not track or not artist_name:
                continue
            if artist_lower not in artist_name.lower():
                continue
            key = f"{track.lower()}|{artist_name.lower()}"
            if key in seen:
                continue
            seen.add(key)
            songs.append((track, artist_name))
            if len(songs) >= 8:
                break

        if not songs:
            return f"No pude validar canciones de '{artist}' con una fuente confiable."

        lines = [f"Canciones de {songs[0][1]} (fuente publica):"]
        for idx, (track, _) in enumerate(songs, start=1):
            lines.append(f"{idx}. {track}")
        return "\n".join(lines)

    async def _tool_weather(self, user_message: str) -> str | None:
        match = re.search(r"(?:clima|tiempo|temperatura)\s+(?:en|de)\s+([a-zA-ZáéíóúñÁÉÍÓÚÑ\s,.-]{2,60})", user_message, re.IGNORECASE)
        if not match:
            return None
        place = match.group(1).strip()

        try:
            with httpx.Client(timeout=8.0) as client:
                geo = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": place, "count": 1, "language": "es", "format": "json"},
                )
                geo.raise_for_status()
                data = geo.json()
                results = data.get("results") or []
                if not results:
                    return f"No pude ubicar '{place}'."
                first = results[0]
                lat = first.get("latitude")
                lon = first.get("longitude")
                city = first.get("name", place)
                country = first.get("country", "")

                weather = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                        "timezone": "auto",
                    },
                )
                weather.raise_for_status()
                wdata = weather.json().get("current", {})
                temp = wdata.get("temperature_2m")
                hum = wdata.get("relative_humidity_2m")
                wind = wdata.get("wind_speed_10m")

                loc = f"{city}, {country}".strip(", ")
                return f"Clima actual en {loc}: {temp}°C, humedad {hum}%, viento {wind} km/h."
        except Exception as exc:
            self.logger.warning("Fallo herramienta clima: %s", exc)
            return None

    def _tool_math(self, user_message: str) -> str | None:
        try:
            import sympy as sp
        except Exception:
            return None

        text = user_message.strip()
        equations = re.findall(r"([A-Za-z0-9\s\+\-\*/\^\(\)\.]+=[A-Za-z0-9\s\+\-\*/\^\(\)\.]+)", text)
        equations = [eq.replace("^", "**") for eq in equations]

        x, y, z = sp.symbols("x y z")
        scope = {"x": x, "y": y, "z": z}

        if equations:
            parsed_eqs = []
            vars_found = set()
            for eq in equations[:3]:
                left, right = eq.split("=", 1)
                left_expr = sp.sympify(left.strip(), locals=scope)
                right_expr = sp.sympify(right.strip(), locals=scope)
                parsed_eqs.append(sp.Eq(left_expr, right_expr))
                vars_found.update(left_expr.free_symbols)
                vars_found.update(right_expr.free_symbols)

            vars_list = [v for v in [x, y, z] if v in vars_found]
            if not vars_list:
                vars_list = [x]

            solution = sp.solve(parsed_eqs, vars_list, dict=True)
            if not solution:
                return "No encontre solucion con esas ecuaciones."
            return f"Solucion: {solution[0]}"

        expr_match = re.search(r"(?:calcula|resolver|resuelve|simplifica)\s+(.+)$", text, re.IGNORECASE)
        if expr_match:
            expr_text = expr_match.group(1).replace("^", "**")
            expr = sp.sympify(expr_text, locals=scope)
            simplified = sp.simplify(expr)
            return f"Resultado: {simplified}"

        return None

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

    def build_text_prompt(
        self,
        user_message: str,
        profile: dict,
        relevant_memory: str,
        history: str,
        style: str = "casual",
        preferences: dict | None = None,
    ) -> str:
        profile_text = json.dumps(profile, ensure_ascii=False)
        short_history = history[-1200:]
        already_started = bool(short_history and short_history != "NINGUNA")
        greeting_rule = "No saludes al usuario en este turno." if already_started else "Puedes saludar brevemente solo en este primer turno."
        style_rule = self._style_instruction(style)
        use_emojis = bool(preferences.get("use_emojis")) if preferences else False
        use_chilean = bool(preferences.get("use_chilean")) if preferences else False
        emoji_rule = (
            "Usa emojis de forma ocasional (maximo 1 por respuesta) cuando aporte calidez."
            if use_emojis
            else "No uses emojis en esta respuesta."
        )
        chilean_rule = (
            "Usa modismos chilenos suaves y naturales, sin exagerar (ej: bacan, al tiro, cachai cuando corresponda)."
            if use_chilean
            else "Mantener espanol neutro, sin modismos regionales marcados."
        )

        return f"""
Eres Zentris, un asistente cercano, util y relajado.

Reglas:
- Responde siempre en espanol.
- {style_rule}
- Responde exactamente a la pregunta actual del usuario.
- No cambies de tema ni inventes contexto.
- Si falta contexto para responder bien, pide una aclaracion breve.
- No inventes datos del usuario.
- Si falta informacion personal, dilo claramente.
- {greeting_rule}
- {emoji_rule}
- {chilean_rule}
- Evita tono legalista o corporativo.
- No repitas frases de relleno ni disclaimers en cada respuesta.
- No hables de privacidad, politicas, limitaciones o almacenamiento salvo que te lo pregunten directo.
- Si no sabes algo, dilo breve y propon una alternativa practica.
- Si el usuario comenta un logro o mejora, reconoce brevemente y sugiere 1-2 ideas accionables para subir de nivel.
- Cuando te pidan recomendaciones, entrega pasos concretos y priorizados (no generalidades).
- Responde de forma breve y accionable, sin repetir informacion ya dicha.

Perfil (solo alta confianza):
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
        final_text = self._strip_internal_leaks(final_text)
        history = self.user_memory.get(user_id, "")
        new_history = history + f"\nUsuario: {user_message}\nZentris: {final_text}"
        self.user_memory[user_id] = new_history[-self.settings.max_history :]
        save_json_atomic(self.settings.memory_file, self.user_memory, self.logger)

    def persist_image_interaction(self, user_id: str, caption: str, reply: str) -> None:
        history = self.user_memory.get(user_id, "")
        new_history = history + f"\nUsuario: [Imagen] {caption if caption else '(sin caption)'}\nZentris: {reply}"
        self.user_memory[user_id] = new_history[-self.settings.max_history :]
        save_json_atomic(self.settings.memory_file, self.user_memory, self.logger)

    def get_profile_summary(self, user_id: str) -> str:
        entry = self.user_profile.get(user_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("data"), dict) or not entry["data"]:
            return "No tengo perfil guardado aun."

        data = entry.get("data", {})
        meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
        style = self.get_user_style(user_id)

        lines = [f"Perfil actual (estilo={style}):"]
        for key in ["nombre", "edad", "city", "nationality", "idioma_preferido", "objetivo", "rol_creador", "studies", "intereses", "gato"]:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list):
                value_text = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                value_text = ", ".join(f"{k}={v}" for k, v in value.items())
            else:
                value_text = str(value)

            info = meta.get(key, {}) if isinstance(meta.get(key), dict) else {}
            confidence = info.get("confidence", 0.0)
            source = info.get("source", "desconocido")
            lines.append(f"- {key}: {value_text} (conf={confidence:.2f}, fuente={source})")

        return "\n".join(lines)

    def clear_profile(self, user_id: str) -> bool:
        existed = user_id in self.user_profile
        if existed:
            self.user_profile.pop(user_id, None)
            save_json_atomic(self.settings.profile_file, self.user_profile, self.logger)
        return existed

    def clear_memory(self, user_id: str) -> bool:
        existed = user_id in self.user_memory
        if existed:
            self.user_memory.pop(user_id, None)
            save_json_atomic(self.settings.memory_file, self.user_memory, self.logger)
        return existed

    async def extract_user_info_async(self, user_id: str, message: str, run_llm: bool = True) -> None:
        current = self.get_profile_data(user_id)

        hinted = self._extract_profile_hints(message, current)
        if hinted:
            self.upsert_profile_data(user_id, hinted, source="mensaje_directo", confidence=0.92)

        if not run_llm:
            return

        if len(message.split()) < 6:
            return

        prompt = f"""
Extrae informacion del usuario en JSON.

Mensaje:
\"{message}\"

Reglas:
- Solo JSON valido
- No inventar
- Solo usa estas claves permitidas: nombre, edad, city, nationality, idioma_preferido, objetivo, rol_creador, studies, intereses, gato
- En gato solo permite: nombre
- Si no hay info: {{}}
"""

        try:
            extracted = await self.ollama.generate_json(prompt, self.settings.fast_model)
            if not extracted:
                return
            cleaned = self._filter_extracted_profile(extracted, message)
            if not cleaned:
                return
            self.upsert_profile_data(user_id, cleaned, source="inferencia", confidence=0.58)
        except Exception as exc:
            self.logger.warning("No se pudo extraer perfil para %s: %s", user_id, exc)

    def extract_user_info_from_image_caption(self, user_id: str, caption: str) -> None:
        if not caption or len(caption.split()) < 2:
            return
        current = self.get_profile_data(user_id)
        hinted = self._extract_profile_hints(caption, current)
        if hinted:
            self.upsert_profile_data(user_id, hinted, source="imagen", confidence=0.75)

    def _extract_profile_hints(self, message: str, current_profile: dict) -> dict:
        msg = message.strip()
        hints: dict = {}

        match = re.search(r"\b(me llamo|mi nombre es|soy)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]{2,30})\b", msg, re.IGNORECASE)
        if match:
            hints["nombre"] = match.group(2).capitalize()
            return hints

        cleaned = re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]", "", msg).strip()
        tokens = [tok for tok in cleaned.split() if tok]
        if len(tokens) == 1 and len(tokens[0]) >= 3:
            if not current_profile.get("nombre"):
                hints["nombre"] = tokens[0].capitalize()

        return hints

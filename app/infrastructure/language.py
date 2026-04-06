import asyncio

from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 4:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_spanish(text: str) -> str:
    return GoogleTranslator(source="auto", target="es").translate(text)


async def enforce_spanish(text: str, balanced_model: str, generate_text_fn, logger) -> str:
    if detect_language(text) == "es":
        return text

    try:
        translated = await asyncio.to_thread(translate_to_spanish, text)
        if translated:
            return translated
    except Exception as exc:
        logger.warning("Fallo traduccion con deep-translator, se usa fallback local: %s", exc)

    translate_prompt = f"""
Reescribe el siguiente texto al espanol natural de forma fiel.
No agregues informacion nueva.

Texto:
{text}
"""
    try:
        translated = await generate_text_fn(translate_prompt, balanced_model)
        return translated if translated else text
    except Exception:
        return text

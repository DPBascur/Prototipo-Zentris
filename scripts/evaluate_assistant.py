import asyncio
import json
import time
from pathlib import Path

from app.application.assistant_core import AssistantCore
from app.config import load_settings
from app.infrastructure.ollama_client import OllamaClient
from app.infrastructure.vector_memory import VectorMemory


def load_eval_prompts(base_dir: Path):
    path = base_dir / "scripts" / "eval_prompts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tone_score(reply: str) -> float:
    text = reply.lower()
    rigid = [
        "como asistente virtual",
        "consentimiento explicito",
        "dentro del alcance permitido",
        "no tengo emociones",
    ]
    return 1.0 if not any(p in text for p in rigid) else 0.0


def precision_score(reply: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    text = reply.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / max(1, len(expected_keywords))


async def run_eval():
    settings = load_settings()
    prompts = load_eval_prompts(settings.base_dir)

    logger = __import__("logging").getLogger("eval")
    logger.setLevel("WARNING")

    ollama = OllamaClient(settings, logger)
    ollama.validate_models()
    vector_memory = VectorMemory(settings, logger)
    core = AssistantCore(settings, logger, ollama, vector_memory)

    user_id = "eval-user"
    rows = []

    for item in prompts:
        prompt = item["prompt"]
        expected = item.get("expected_keywords", [])

        started = time.perf_counter()
        tool_reply = await core.try_tool_response(prompt)
        if tool_reply:
            prefs = core.update_response_preferences(user_id, prompt)
            history = core.user_memory.get(user_id, "")
            reply = core.finalize_reply(tool_reply, prefs, user_message=prompt, history=history)
        else:
            profile = core.get_profile_for_prompt(user_id)
            history = core.user_memory.get(user_id, "")
            memory = core.vector_memory.get_relevant_memories(user_id, prompt) if len(prompt.split()) > 2 else ""
            style = core.get_user_style(user_id)
            prefs = core.update_response_preferences(user_id, prompt)
            model = core.choose_model(prompt)
            llm_prompt = core.build_text_prompt(prompt, profile, memory, history, style, prefs)
            reply = await ollama.generate_text(llm_prompt, model)
            if reply and not core.is_reply_on_topic(prompt, reply):
                repair = core.build_precision_prompt(prompt, profile, memory, history, style, prefs)
                reply = await ollama.generate_text(repair, settings.balanced_model)
            reply = core.finalize_reply(reply, prefs, user_message=prompt, history=history)

        elapsed_ms = (time.perf_counter() - started) * 1000
        coherent = 1.0 if core.is_reply_on_topic(prompt, reply) else 0.0
        precise = precision_score(reply, expected)
        tone = tone_score(reply)

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt,
                "coherencia": coherent,
                "precision": round(precise, 3),
                "tono": tone,
                "latencia_ms": round(elapsed_ms, 1),
                "respuesta": reply,
            }
        )

        core.persist_text_interaction(user_id, prompt, reply)

    avg_coh = sum(r["coherencia"] for r in rows) / len(rows)
    avg_pre = sum(r["precision"] for r in rows) / len(rows)
    avg_tone = sum(r["tono"] for r in rows) / len(rows)
    avg_lat = sum(r["latencia_ms"] for r in rows) / len(rows)

    report = {
        "total": len(rows),
        "promedio": {
            "coherencia": round(avg_coh, 3),
            "precision": round(avg_pre, 3),
            "tono": round(avg_tone, 3),
            "latencia_ms": round(avg_lat, 1),
        },
        "resultados": rows,
    }

    out_path = settings.base_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluacion completada: {out_path}")
    print(json.dumps(report["promedio"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_eval())

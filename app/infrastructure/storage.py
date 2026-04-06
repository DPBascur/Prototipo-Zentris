import json
import os
import tempfile
from pathlib import Path


def load_json_or_default(path: Path, default: dict, logger) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        logger.warning("%s no contiene un objeto JSON valido. Se usa {}.", path.name)
    except FileNotFoundError:
        logger.info("%s no existe aun. Se crea en el primer guardado.", path.name)
    except Exception as exc:
        logger.warning("No se pudo leer %s: %s", path.name, exc)
    return default


def save_json_atomic(path: Path, data: dict, logger) -> None:
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    except Exception as exc:
        logger.warning("No se pudo guardar %s: %s", path.name, exc)

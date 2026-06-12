import json
from pathlib import Path
from datetime import datetime


def ensure_json_file(path: Path, default_data):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        save_json(path, default_data)


def load_json(path: Path, default_data):
    ensure_json_file(path, default_data)

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Warning: {path} was corrupted. Resetting to default.")
        save_json(path, default_data)
        return default_data


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def now_timestamp():
    return datetime.now().isoformat(timespec="seconds")
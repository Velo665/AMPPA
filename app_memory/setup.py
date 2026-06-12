from config import (
    MEMORY_DIR,
    ENTITIES_DIR,
    CONVERSATION_FILE,
    USER_PROFILE_FILE,
    PEOPLE_FILE,
    PROJECTS_FILE,
    CONCEPTS_FILE
)

from app_memory.json_store import (
    ensure_json_file,
)


# -----------------------------
# Setup memory files
# -----------------------------

def ensure_memory_files():
    MEMORY_DIR.mkdir(exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_json_file(CONVERSATION_FILE, [])
    ensure_json_file(USER_PROFILE_FILE, {
        "name": "Jarell",
        "preferred_name": "Max",
        "active_projects": [],
        "long_term_goals": [],
        "preferences": [],
        "learning_style": [],
        "assistant_rules": [],
        "current_focus": [],
        "last_updated": None
    })

    ensure_json_file(PEOPLE_FILE, {})
    ensure_json_file(PROJECTS_FILE, {})
    ensure_json_file(CONCEPTS_FILE, {})

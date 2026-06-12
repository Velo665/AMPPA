from pathlib import Path


# -----------------------------
# File paths
# -----------------------------

MEMORY_DIR = Path("memory")
ENTITIES_DIR = MEMORY_DIR / "entities"

CONVERSATION_FILE = MEMORY_DIR / "conversation.json"
USER_PROFILE_FILE = MEMORY_DIR / "user_profile.json"

PEOPLE_FILE = ENTITIES_DIR / "people.json"
PROJECTS_FILE = ENTITIES_DIR / "projects.json"
CONCEPTS_FILE = ENTITIES_DIR / "concepts.json"


# -----------------------------
# Memory settings
# -----------------------------

MAX_MEMORY_MESSAGES = 10


# -----------------------------
# LM Studio settings
# -----------------------------

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"
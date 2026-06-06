import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI


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

MAX_MEMORY_MESSAGES = 10


# -----------------------------
# LM Studio client
# -----------------------------

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


# -----------------------------
# Generic JSON helpers
# -----------------------------

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


# -----------------------------
# Setup memory files
# -----------------------------

def ensure_memory_files():
    MEMORY_DIR.mkdir(exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_json_file(CONVERSATION_FILE, [])
    ensure_json_file(USER_PROFILE_FILE, {
        "name": "Jarell",
        "goals": [],
        "preferences": [],
        "projects": []
    })

    ensure_json_file(PEOPLE_FILE, {})
    ensure_json_file(PROJECTS_FILE, {})
    ensure_json_file(CONCEPTS_FILE, {})


# -----------------------------
# Conversation memory
# -----------------------------

def load_conversation_memory():
    return load_json(CONVERSATION_FILE, [])


def save_conversation_memory(memory):
    save_json(CONVERSATION_FILE, memory)


def add_to_conversation_memory(user_message, amppa_response):
    memory = load_conversation_memory()

    memory.append({
        "timestamp": now_timestamp(),
        "user": user_message,
        "amppa": amppa_response
    })

    save_conversation_memory(memory)


def build_recent_conversation_context():
    memory = load_conversation_memory()
    recent_memory = memory[-MAX_MEMORY_MESSAGES:]

    if not recent_memory:
        return "No prior conversation memory yet."

    context_lines = []

    for item in recent_memory:
        context_lines.append(f"User: {item['user']}")
        context_lines.append(f"AMPPA: {item['amppa']}")

    return "\n".join(context_lines)


def show_conversation_memory():
    memory = load_conversation_memory()

    if not memory:
        print("\nAMPPA: Conversation memory is empty.")
        return

    print("\n--- Recent Conversation Memory ---")
    for item in memory[-MAX_MEMORY_MESSAGES:]:
        print(f"\n[{item['timestamp']}]")
        print(f"You: {item['user']}")
        print(f"AMPPA: {item['amppa']}")
    print("\n--- End Memory ---")


def clear_conversation_memory():
    save_conversation_memory([])
    print("\nAMPPA: Conversation memory cleared.")


# -----------------------------
# Entity memory helpers
# -----------------------------

def get_entity_file(entity_type: str) -> Path:
    entity_type = entity_type.lower()

    if entity_type == "person":
        return PEOPLE_FILE

    if entity_type == "project":
        return PROJECTS_FILE

    if entity_type == "concept":
        return CONCEPTS_FILE

    raise ValueError(f"Unknown entity type: {entity_type}")


def load_entities(entity_type: str):
    file_path = get_entity_file(entity_type)
    return load_json(file_path, {})


def save_entities(entity_type: str, entities):
    file_path = get_entity_file(entity_type)
    save_json(file_path, entities)


def normalize_entity_name(name: str) -> str:
    return name.strip()


def remember_entity(entity_type: str, name: str, fact: str):
    name = normalize_entity_name(name)
    fact = fact.strip()

    if not name or not fact:
        print("\nAMPPA: Use this format: /remember-person Name | fact")
        return

    entities = load_entities(entity_type)

    if name not in entities:
        entities[name] = {
            "type": entity_type,
            "known_facts": [],
            "created_at": now_timestamp(),
            "last_mentioned": now_timestamp(),
            "mention_count": 0
        }

    if fact not in entities[name]["known_facts"]:
        entities[name]["known_facts"].append(fact)

    entities[name]["last_mentioned"] = now_timestamp()
    entities[name]["mention_count"] = entities[name].get("mention_count", 0) + 1

    save_entities(entity_type, entities)

    print(f"\nAMPPA: Remembered {entity_type} '{name}'.")


def get_entity(entity_type: str, name: str):
    name = normalize_entity_name(name)
    entities = load_entities(entity_type)

    if name not in entities:
        print(f"\nAMPPA: I do not have memory for {entity_type} '{name}' yet.")
        return

    entity = entities[name]

    print(f"\n--- {entity_type.title()}: {name} ---")
    print(f"Created: {entity.get('created_at', 'Unknown')}")
    print(f"Last mentioned: {entity.get('last_mentioned', 'Unknown')}")
    print(f"Mention count: {entity.get('mention_count', 0)}")

    facts = entity.get("known_facts", [])

    if facts:
        print("\nKnown facts:")
        for index, fact in enumerate(facts, start=1):
            print(f"{index}. {fact}")
    else:
        print("\nNo known facts saved yet.")

    print("--- End Entity ---")


def list_entities():
    people = load_entities("person")
    projects = load_entities("project")
    concepts = load_entities("concept")

    print("\n--- Entity Memory ---")

    print("\nPeople:")
    if people:
        for name in people:
            print(f"- {name}")
    else:
        print("- None")

    print("\nProjects:")
    if projects:
        for name in projects:
            print(f"- {name}")
    else:
        print("- None")

    print("\nConcepts:")
    if concepts:
        for name in concepts:
            print(f"- {name}")
    else:
        print("- None")

    print("\n--- End Entities ---")


def build_entity_context():
    people = load_entities("person")
    projects = load_entities("project")
    concepts = load_entities("concept")

    lines = []

    def add_entity_section(title, entities):
        if not entities:
            return

        lines.append(f"\n{title}:")
        for name, data in entities.items():
            facts = data.get("known_facts", [])
            if facts:
                recent_facts = facts[-3:]
                fact_text = "; ".join(recent_facts)
                lines.append(f"- {name}: {fact_text}")

    add_entity_section("Known people", people)
    add_entity_section("Known projects", projects)
    add_entity_section("Known concepts", concepts)

    if not lines:
        return "No structured entity memory yet."

    return "\n".join(lines).strip()


# -----------------------------
# Command parsing
# -----------------------------

def parse_remember_command(user_message: str, command: str):
    content = user_message[len(command):].strip()

    if "|" not in content:
        print(f"\nAMPPA: Use this format: {command} Name | fact")
        return None, None

    name, fact = content.split("|", 1)
    return name.strip(), fact.strip()


def handle_command(user_message: str) -> bool:
    lower_message = user_message.lower()

    if lower_message == "/memory":
        show_conversation_memory()
        return True

    if lower_message == "/clear-memory":
        clear_conversation_memory()
        return True

    if lower_message == "/entities":
        list_entities()
        return True

    if lower_message.startswith("/remember-person "):
        name, fact = parse_remember_command(user_message, "/remember-person")
        if name and fact:
            remember_entity("person", name, fact)
        return True

    if lower_message.startswith("/remember-project "):
        name, fact = parse_remember_command(user_message, "/remember-project")
        if name and fact:
            remember_entity("project", name, fact)
        return True

    if lower_message.startswith("/remember-concept "):
        name, fact = parse_remember_command(user_message, "/remember-concept")
        if name and fact:
            remember_entity("concept", name, fact)
        return True

    if lower_message.startswith("/person "):
        name = user_message[len("/person"):].strip()
        get_entity("person", name)
        return True

    if lower_message.startswith("/project "):
        name = user_message[len("/project"):].strip()
        get_entity("project", name)
        return True

    if lower_message.startswith("/concept "):
        name = user_message[len("/concept"):].strip()
        get_entity("concept", name)
        return True

    if lower_message == "/help":
        show_help()
        return True

    return False


def show_help():
    print("""
--- AMPPA Commands ---

Conversation memory:
/memory
/clear-memory

Entity memory:
/remember-person Name | fact
/remember-project Name | fact
/remember-concept Name | fact

View entity memory:
/person Name
/project Name
/concept Name
/entities

Other:
exit
quit
/help

--- End Help ---
""")


# -----------------------------
# AMPPA chat
# -----------------------------

def ask_amppa(user_message):
    recent_conversation_context = build_recent_conversation_context()
    entity_context = build_entity_context()

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AMPPA, a private local AI assistant built by Jarell. "
                    "You are honest, direct, useful, and focused on learning, discipline, "
                    "memory, privacy, and long-term growth. "
                    "You are currently running locally through LM Studio. "
                    "Use the provided memory when relevant, but do not pretend to remember "
                    "things that are not in memory."
                )
            },
            {
                "role": "system",
                "content": f"Recent conversation memory:\n{recent_conversation_context}"
            },
            {
                "role": "system",
                "content": f"Structured entity memory:\n{entity_context}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.7,
        max_tokens=400,
    )

    amppa_response = response.choices[0].message.content.strip()

    add_to_conversation_memory(user_message, amppa_response)

    return amppa_response


# -----------------------------
# Main app loop
# -----------------------------

def main():
    ensure_memory_files()

    print("AMPPA is connected to LM Studio locally.")
    print("Local memory is enabled.")
    print("Structured entity memory is enabled.")
    print("Type 'exit' to quit.")
    print("Type '/help' to see commands.")

    while True:
        user_message = input("\nYou: ").strip()

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("AMPPA: Goodbye.")
            break

        if handle_command(user_message):
            continue

        response = ask_amppa(user_message)
        print(f"\nAMPPA: {response}")


if __name__ == "__main__":
    main()
from openai import OpenAI
from pathlib import Path

# configuration and constants
from config import (
    MEMORY_DIR,
    ENTITIES_DIR,
    CONVERSATION_FILE,
    USER_PROFILE_FILE,
    PEOPLE_FILE,
    PROJECTS_FILE,
    CONCEPTS_FILE,
    MAX_MEMORY_MESSAGES,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_MODEL,
)

from app_memory.json_store import (
    ensure_json_file,
    load_json,
    save_json,
    now_timestamp,
)

# -----------------------------
# LM Studio client
# -----------------------------

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)


# -----------------------------
# User Profile memory
# -----------------------------

VALID_PROFILE_FIELDS = {
    "active_projects",
    "long_term_goals",
    "preferences",
    "learning_style",
    "assistant_rules",
    "current_focus"
}

def load_user_profile():
    return load_json(USER_PROFILE_FILE, {
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

def save_user_profile(profile):
    profile["last_updated"] = now_timestamp()
    save_json(USER_PROFILE_FILE, profile)

def remember_profile_fact(field: str, fact: str):
    field = field.strip().casefold().replace("-", "_")
    fact = fact.strip()

    if field not in VALID_PROFILE_FIELDS:
        print("\nAMPPA: Unknown profile field.")
        print("Valid fields are:")
        for valid_field in sorted(VALID_PROFILE_FIELDS):
            print(f"- {valid_field}")
        return
    
    if not fact:
        print("\nAMPPA: Use this format: /profile-add field | fact")
        return

    profile = load_user_profile()

    if field not in profile or not isinstance(profile[field], list):
        profile[field] = []

    if fact not in profile[field]:
        profile[field].append(fact)

    save_user_profile(profile)

    print(f"\nAMPPA: Remembered profile fact in '{field}'.")

def show_user_profile():
    profile = load_user_profile()

    print("\n--- User Profile ---")
    print(f"Name: {profile.get('name', 'Unknown')}")
    print(f"Preferred name: {profile.get('preferred_name', 'Unknown')}")
    print(f"Last updated: {profile.get('last_updated', 'Unknown')}")

    for field in [
        "active_projects",
        "long_term_goals",
        "preferences",
        "learning_style",
        "assistant_rules",
        "current_focus"
    ]:
        values = profile.get(field, [])

        print(f"\n{field.replace('_', ' ').title()}:")
        if values:
            for item in values:
                print(f"- {item}")
        else:
            print("- None")
    
    print("\n--- End Profile ---")
    

def build_user_profile_context():
    profile = load_user_profile()

    lines = [
        f"Name: {profile.get('name', 'Unknown')}",
        f"Preferred name: {profile.get('preferred_name', 'Unknown')}"
    ]

    for field in [
        "active_projects",
        "long_term_goals",
        "preferences",
        "learning_style",
        "assistant_rules",
        "current_focus"
    ]:
        values = profile.get(field, [])
        if values:
            lines.append(f"\n{field.replace('_', ' ').title()}:")
            for item in values:
                lines.append(f"- {item}")

    return "\n".join(lines)


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

    existing_key = find_entity_key(entities, name)

    if existing_key:
        entity_key = existing_key
    else:
        entity_key = name
        entities[entity_key]= {
            "type": entity_type,
            "display_name": name,
            "known_facts": [],
            "created_at": now_timestamp(),
            "last_mentioned": now_timestamp(),
            "mention_count": 0
        }

    if fact not in entities[entity_key]["known_facts"]:
        entities[entity_key]["known_facts"].append(fact)

    entities[entity_key]["last_mentioned"] = now_timestamp()
    entities[entity_key]["mention_count"] = entities[entity_key].get("mention_count", 0) + 1

    save_entities(entity_type, entities)

    print(f"\nAMPPA: Remembered {entity_type} '{name}'.")


def get_entity(entity_type: str, name: str):
    name = normalize_entity_name(name)
    entities = load_entities(entity_type)

    entity_key = find_entity_key(entities, name)

    if not entity_key:
        print(f"\nAMPPA: I do not have memory for {entity_type} '{name}' yet.")
        return

    entity = entities[entity_key]

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

# entity key for normalization and lookup, but we can display the original name with proper capitalization when listing or showing details.

def find_entity_key(entities, name: str):
    search_name = name.strip().casefold()

    for existing_name in entities:
        if existing_name.casefold() == search_name:
            return existing_name
    return None


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
    
    if lower_message == "/profile":
        show_user_profile()
        return True
    
    if lower_message.startswith("/profile-add "):
        content = user_message[len("/profile-add "):].strip()

        if "|" not in content:
            print("\nAMPPA: Use this format: /profile-add field | fact")
            return True

        field, fact = content.split("|", 1)
        remember_profile_fact(field.strip(), fact.strip())
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
          
/profile
/profile-add field | fact

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
    user_profile_context = build_user_profile_context()

    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
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
                "role": "system",
                "content": f"User profile:\n{user_profile_context}"
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
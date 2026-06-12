from openai import OpenAI

from app_memory.user_profile import (
    remember_profile_fact,
    show_user_profile,
    build_user_profile_context,
)

from app_memory.entity_memory import (
    remember_entity,
    get_entity,
    list_entities,
    build_entity_context,
)

from app_memory.conversation_memory import (
    add_to_conversation_memory,
    build_recent_conversation_context,
    show_conversation_memory,
    clear_conversation_memory,
)

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
)

# -----------------------------
# LM Studio client
# -----------------------------

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
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
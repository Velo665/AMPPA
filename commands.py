# -----------------------------
# Command parsing
# -----------------------------

from app_memory.user_profile import (
    remember_profile_fact,
    show_user_profile,
)

from app_memory.entity_memory import (
    remember_entity,
    get_entity,
    list_entities,
)

from app_memory.conversation_memory import (
    show_conversation_memory,
    clear_conversation_memory,
)


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
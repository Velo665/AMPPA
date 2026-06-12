from config import (
    CONVERSATION_FILE,
    MAX_MEMORY_MESSAGES,
)

from app_memory.json_store import (
    load_json,
    save_json,
    now_timestamp,
)

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

import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI


MEMORY_DIR = Path("memory")
CONVERSATION_FILE = MEMORY_DIR / "conversation.json"

MAX_MEMORY_MESSAGES = 10


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def ensure_memory_file():
    MEMORY_DIR.mkdir(exist_ok=True)

    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.write_text("[]", encoding="utf-8")


def load_conversation_memory():
    ensure_memory_file()

    try:
        with CONVERSATION_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("Warning: conversation memory file was corrupted. Starting with empty memory.")
        return []


def save_conversation_memory(memory):
    ensure_memory_file()

    with CONVERSATION_FILE.open("w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2)


def add_to_memory(user_message, amppa_response):
    memory = load_conversation_memory()

    memory.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": user_message,
        "amppa": amppa_response
    })

    save_conversation_memory(memory)


def build_recent_context(memory):
    recent_memory = memory[-MAX_MEMORY_MESSAGES:]

    if not recent_memory:
        return "No prior conversation memory yet."

    context_lines = []

    for item in recent_memory:
        context_lines.append(f"User: {item['user']}")
        context_lines.append(f"AMPPA: {item['amppa']}")

    return "\n".join(context_lines)


def ask_amppa(user_message):
    memory = load_conversation_memory()
    recent_context = build_recent_context(memory)

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
                    "Use the recent conversation memory when it is relevant, but do not "
                    "pretend to remember things that are not in memory."
                )
            },
            {
                "role": "system",
                "content": f"Recent conversation memory:\n{recent_context}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )

    amppa_response = response.choices[0].message.content.strip()

    add_to_memory(user_message, amppa_response)

    return amppa_response


def show_memory():
    memory = load_conversation_memory()

    if not memory:
        print("\nAMPPA: Memory is empty.")
        return

    print("\n--- Recent Memory ---")
    for item in memory[-MAX_MEMORY_MESSAGES:]:
        print(f"\n[{item['timestamp']}]")
        print(f"You: {item['user']}")
        print(f"AMPPA: {item['amppa']}")
    print("\n--- End Memory ---")


def clear_memory():
    save_conversation_memory([])
    print("\nAMPPA: Conversation memory cleared.")


def main():
    ensure_memory_file()

    print("AMPPA is connected to LM Studio locally.")
    print("Local memory is enabled.")
    print("Type 'exit' to quit.")
    print("Commands: /memory, /clear-memory")

    while True:
        user_message = input("\nYou: ").strip()

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("AMPPA: Goodbye.")
            break

        if user_message.lower() == "/memory":
            show_memory()
            continue

        if user_message.lower() == "/clear-memory":
            clear_memory()
            continue

        response = ask_amppa(user_message)
        print(f"\nAMPPA: {response}")


if __name__ == "__main__":
    main()
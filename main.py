from openai import OpenAI

from commands import (
    handle_command,
)

from app_memory.setup import (
    ensure_memory_files,
)

from app_memory.user_profile import (
    build_user_profile_context,
)

from app_memory.entity_memory import (
    build_entity_context,
)

from app_memory.conversation_memory import (
    add_to_conversation_memory,
    build_recent_conversation_context,
)

# configuration and constants
from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_MODEL,
)

# -----------------------------
# LM Studio client
# -----------------------------

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)


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
from app_memory.user_profile import build_user_profile_context
from app_memory.entity_memory import build_entity_context
from app_memory.conversation_memory import build_recent_conversation_context


def build_amppa_messages(user_message: str):
    recent_conversation_context = build_recent_conversation_context()
    entity_context = build_entity_context()
    user_profile_context = build_user_profile_context()

    return [
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
    ]
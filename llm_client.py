from openai import OpenAI

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_MODEL,
)

from app_memory.conversation_memory import add_to_conversation_memory
from prompt_builder import build_amppa_messages

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)


def ask_amppa(user_message):
    response = client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=build_amppa_messages(user_message),
        temperature=0.7,
        max_tokens=400,
    )

    amppa_response = response.choices[0].message.content.strip()

    add_to_conversation_memory(user_message, amppa_response)

    return amppa_response

def check_lm_studio_status():
    try:
        models = client.models.list()

        if not models.data:
            print("AMPPA warning: LM Studio is reachable, but no model is loaded.")
            print("Load a model in LM Studio before chatting.")
            return False

        print(f"LM Studio model loaded: {models.data[0].id}")
        return True

    except Exception as error:
        print("AMPPA warning: Could not reach LM Studio.")
        print("Make sure the LM Studio local server is running.")
        print(f"Details: {error}")
        return False
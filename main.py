from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def ask_amppa(user_message: str) -> str:
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AMPPA, a private local AI assistant built by Jarell. "
                    "You are honest, direct, useful, and focused on learning, "
                    "discipline, memory, and long-term growth."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def main():
    print("AMPPA is connected to LM Studio locally.")
    print("Type 'exit' to quit.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"exit", "quit"}:
            print("AMPPA: Goodbye.")
            break

        response = ask_amppa(user_message)
        print(f"\nAMPPA: {response}")


if __name__ == "__main__":
    main()
from app_memory.setup import ensure_memory_files
from commands import handle_command
from llm_client import ask_amppa


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
from config import (
    USER_PROFILE_FILE,
)
from app_memory.json_store import (
    load_json,
    save_json,
    now_timestamp,
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

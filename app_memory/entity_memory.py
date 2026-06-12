from pathlib import Path

from app_memory.json_store import (
    load_json,
    save_json,
    now_timestamp,
)

from config import (
    PEOPLE_FILE,
    PROJECTS_FILE,
    CONCEPTS_FILE,
)

# -----------------------------
# Entity memory helpers
# -----------------------------

def get_entity_file(entity_type: str) -> Path:
    entity_type = entity_type.lower()

    if entity_type == "person":
        return PEOPLE_FILE

    if entity_type == "project":
        return PROJECTS_FILE

    if entity_type == "concept":
        return CONCEPTS_FILE

    raise ValueError(f"Unknown entity type: {entity_type}")


def load_entities(entity_type: str):
    file_path = get_entity_file(entity_type)
    return load_json(file_path, {})


def save_entities(entity_type: str, entities):
    file_path = get_entity_file(entity_type)
    save_json(file_path, entities)


def normalize_entity_name(name: str) -> str:
    return name.strip()


def remember_entity(entity_type: str, name: str, fact: str):
    name = normalize_entity_name(name)
    fact = fact.strip()

    if not name or not fact:
        print("\nAMPPA: Use this format: /remember-person Name | fact")
        return

    entities = load_entities(entity_type)

    existing_key = find_entity_key(entities, name)

    if existing_key:
        entity_key = existing_key
    else:
        entity_key = name
        entities[entity_key]= {
            "type": entity_type,
            "display_name": name,
            "known_facts": [],
            "created_at": now_timestamp(),
            "last_mentioned": now_timestamp(),
            "mention_count": 0
        }

    if fact not in entities[entity_key]["known_facts"]:
        entities[entity_key]["known_facts"].append(fact)

    entities[entity_key]["last_mentioned"] = now_timestamp()
    entities[entity_key]["mention_count"] = entities[entity_key].get("mention_count", 0) + 1

    save_entities(entity_type, entities)

    print(f"\nAMPPA: Remembered {entity_type} '{name}'.")


def get_entity(entity_type: str, name: str):
    name = normalize_entity_name(name)
    entities = load_entities(entity_type)

    entity_key = find_entity_key(entities, name)

    if not entity_key:
        print(f"\nAMPPA: I do not have memory for {entity_type} '{name}' yet.")
        return

    entity = entities[entity_key]

    print(f"\n--- {entity_type.title()}: {name} ---")
    print(f"Created: {entity.get('created_at', 'Unknown')}")
    print(f"Last mentioned: {entity.get('last_mentioned', 'Unknown')}")
    print(f"Mention count: {entity.get('mention_count', 0)}")

    facts = entity.get("known_facts", [])

    if facts:
        print("\nKnown facts:")
        for index, fact in enumerate(facts, start=1):
            print(f"{index}. {fact}")
    else:
        print("\nNo known facts saved yet.")

    print("--- End Entity ---")

# entity key for normalization and lookup, but we can display the original name with proper capitalization when listing or showing details.

def find_entity_key(entities, name: str):
    search_name = name.strip().casefold()

    for existing_name in entities:
        if existing_name.casefold() == search_name:
            return existing_name
    return None


def list_entities():
    people = load_entities("person")
    projects = load_entities("project")
    concepts = load_entities("concept")

    print("\n--- Entity Memory ---")

    print("\nPeople:")
    if people:
        for name in people:
            print(f"- {name}")
    else:
        print("- None")

    print("\nProjects:")
    if projects:
        for name in projects:
            print(f"- {name}")
    else:
        print("- None")

    print("\nConcepts:")
    if concepts:
        for name in concepts:
            print(f"- {name}")
    else:
        print("- None")

    print("\n--- End Entities ---")


def build_entity_context():
    people = load_entities("person")
    projects = load_entities("project")
    concepts = load_entities("concept")

    lines = []

    def add_entity_section(title, entities):
        if not entities:
            return

        lines.append(f"\n{title}:")
        for name, data in entities.items():
            facts = data.get("known_facts", [])
            if facts:
                recent_facts = facts[-3:]
                fact_text = "; ".join(recent_facts)
                lines.append(f"- {name}: {fact_text}")

    add_entity_section("Known people", people)
    add_entity_section("Known projects", projects)
    add_entity_section("Known concepts", concepts)

    if not lines:
        return "No structured entity memory yet."

    return "\n".join(lines).strip()


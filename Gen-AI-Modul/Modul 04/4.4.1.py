import json
from pathlib import Path


def save_conversation(history: list[dict], path: str) -> None:
    """Serialize conversation history to a JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_conversation(path: str) -> list[dict]:
    """Deserialize conversation history from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sample_history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation."},
    ]

    save_conversation(sample_history, "data/chat_history.json")
    print("Saved conversation to data/chat_history.json")

    loaded = load_conversation("data/chat_history.json")
    print("Loaded back:")
    print(loaded)
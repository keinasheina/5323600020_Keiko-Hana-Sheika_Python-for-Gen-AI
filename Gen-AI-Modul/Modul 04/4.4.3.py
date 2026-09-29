import json
import os
from pathlib import Path
from typing import Any


def load_config(config_path: str, env_prefix: str = "") -> dict[str, Any]:
    """Load a JSON config file, then override values with matching env vars."""
    path = Path(config_path)
    config: dict[str, Any] = {}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))

    for key in config:
        env_key = f"{env_prefix}{key}".upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            config[key] = _coerce_type(env_value, config[key])

    return config


def _coerce_type(value: str, reference: Any) -> Any:
    """Coerce an env var string to match the type of the existing config value."""
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


if __name__ == "__main__":
    # simulasikan env var override
    os.environ["TEMPERATURE"] = "0.7"
    os.environ["DEBUG"] = "true"

    result = load_config("config.json")
    print(result)
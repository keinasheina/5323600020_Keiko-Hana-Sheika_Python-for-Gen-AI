import csv
import threading
from datetime import datetime, timezone
from pathlib import Path


class LLMCallLogger:
    """Thread-safe CSV logger for LLM call metrics."""

    _HEADERS = ["timestamp", "model", "input_tokens", "output_tokens", "latency_ms"]

    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._HEADERS)

    def log(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            model,
            input_tokens,
            output_tokens,
            round(latency_ms, 2),
        ]
        with self._lock:
            with self._path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)


if __name__ == "__main__":
    logger = LLMCallLogger("logs/llm_calls.csv")

    logger.log(model="claude-sonnet-4-5", input_tokens=120, output_tokens=340, latency_ms=512.7)
    logger.log(model="gpt-4o", input_tokens=98, output_tokens=280, latency_ms=430.2)

    print("Logged 2 calls to logs/llm_calls.csv")

    # baca ulang buat verifikasi isinya
    with open("logs/llm_calls.csv", encoding="utf-8") as f:
        print(f.read())
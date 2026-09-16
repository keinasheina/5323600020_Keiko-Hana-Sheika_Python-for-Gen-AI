import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    prompt: str
    model: str
    response: str
    latency_ms: float
    score: float = field(default=0.0)


def mock_llm_call(prompt: str, model: str) -> tuple[str, float]:
    """Simulate an LLM call. Returns (response, latency_ms)."""

    rng = np.random.default_rng(
        abs(hash(prompt + model)) % 2**31
    )

    latency = rng.uniform(300, 700)

    answers = {
        "What is a transformer?":
            "A transformer is a deep learning model that uses "
            "attention mechanisms to process sequential data.",

        "Define RAG":
            "RAG stands for Retrieval-Augmented Generation. "
            "It combines information retrieval with text generation.",

        "What is fine-tuning?":
            "Fine-tuning is additional training that adjusts "
            "the model weights for a specific task."
    }

    response = f"[{model}] {answers.get(prompt, 'No answer available.')}"

    return response, latency


def score_response(
    response: str,
    expected_keywords: list[str]
) -> float:
    """Simple keyword-based scorer (0.0 - 1.0)."""

    if not expected_keywords:
        return 0.0

    found = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in response.lower()
    )

    return found / len(expected_keywords)


# ==========================================
# Dataset Evaluasi
# ==========================================

prompts = [
    ("What is a transformer?", ["attention", "model"]),
    ("Define RAG", ["retrieval", "generation"]),
    ("What is fine-tuning?", ["training", "weights"]),
]


# ==========================================
# Model yang akan dievaluasi
# ==========================================

models = [
    "claude-sonnet-4-5",
    "gpt-4o"
]


# ==========================================
# Menjalankan Evaluasi
# ==========================================

results: list[EvalResult] = []

for prompt, keywords in prompts:
    for model in models:

        response, latency = mock_llm_call(
            prompt,
            model
        )

        score = score_response(
            response,
            keywords
        )

        results.append(
            EvalResult(
                prompt=prompt,
                model=model,
                response=response,
                latency_ms=latency,
                score=score
            )
        )


# ==========================================
# Analisis dengan Pandas
# ==========================================

df = pd.DataFrame(
    [vars(result) for result in results]
)

summary = (
    df.groupby("model")
    .agg(
        avg_score=("score", "mean"),
        avg_latency=("latency_ms", "mean")
    )
    .round(3)
)

print("\n=== Evaluation Summary ===")
print(summary)

print("\n=== Detail Results ===")
print(df.to_string(index=False))


# ==========================================
# Simpan Hasil
# ==========================================

# Simpan di folder yang sama dengan file Python
output_file = Path(__file__).parent / "eval_results.csv"

df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")
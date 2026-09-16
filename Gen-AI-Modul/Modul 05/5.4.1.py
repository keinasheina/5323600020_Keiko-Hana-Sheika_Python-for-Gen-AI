import pandas as pd
from pathlib import Path


# ==========================================
# Membuat / membaca file CSV benchmark
# ==========================================

BASE_DIR = Path(__file__).parent
csv_file = BASE_DIR / "llm_benchmark.csv"


# Buat CSV jika belum ada
if not csv_file.exists():

    data = [
        ["GPT-4o", "Question Answering", 0.91, 420],
        ["GPT-4o", "Summarization", 0.88, 450],
        ["GPT-4o", "Classification", 0.94, 390],
        ["GPT-4o", "Reasoning", 0.89, 520],
        ["GPT-4o", "Translation", 0.93, 410],

        ["Claude Sonnet", "Question Answering", 0.93, 450],
        ["Claude Sonnet", "Summarization", 0.92, 470],
        ["Claude Sonnet", "Classification", 0.91, 430],
        ["Claude Sonnet", "Reasoning", 0.94, 500],
        ["Claude Sonnet", "Translation", 0.95, 440],

        ["Gemini 1.5", "Question Answering", 0.89, 380],
        ["Gemini 1.5", "Summarization", 0.90, 400],
        ["Gemini 1.5", "Classification", 0.92, 370],
        ["Gemini 1.5", "Reasoning", 0.87, 460],
        ["Gemini 1.5", "Translation", 0.91, 390],

        ["Llama 3", "Question Answering", 0.86, 330],
        ["Llama 3", "Summarization", 0.85, 350],
        ["Llama 3", "Classification", 0.88, 320],
        ["Llama 3", "Reasoning", 0.84, 410],
        ["Llama 3", "Translation", 0.87, 340],
    ]

    df = pd.DataFrame(
        data,
        columns=["model", "task", "score", "latency"]
    )

    df.to_csv(csv_file, index=False)

    print(f"CSV dibuat: {csv_file}")


# ==========================================
# Load CSV
# ==========================================

df = pd.read_csv(csv_file)

print("\n=== DATA BENCHMARK ===")
print(df.to_string(index=False))


# ==========================================
# Mean score per model
# ==========================================

mean_score = df.groupby("model")["score"].mean()

print("\n=== MEAN SCORE PER MODEL ===")
print(mean_score.round(3))


# ==========================================
# Best-performing task per model
# ==========================================

best_task = df.loc[
    df.groupby("model")["score"].idxmax(),
    ["model", "task", "score"]
]

print("\n=== BEST-PERFORMING TASK ===")
print(best_task.to_string(index=False))


# ==========================================
# Correlation score dan latency
# ==========================================

correlation = df["score"].corr(df["latency"])

print(
    f"\n=== CORRELATION ===\n"
    f"Correlation score-latency: {correlation:.3f}"
)

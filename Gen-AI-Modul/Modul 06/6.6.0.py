import anthropic, os
from dotenv import load_dotenv

load_dotenv()

# Klien disesuaikan dengan base_url baru
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

# 1. Menghitung token menggunakan model dan pesan yang baru
response_tokens = client.messages.count_tokens(
    model="qwen/qwen3.8-max:free",
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)
print(f"Estimated input tokens:{response_tokens.input_tokens}")

# 2. Estimator Biaya
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00}, # per 1M tokens
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "qwen/qwen3.8-max:free": {"input": 0.00,  "output": 0.00}, # Ditambahkan untuk model free
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model:{model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

cost = estimate_cost("qwen/qwen3.8-max:free", input_tokens=response_tokens.input_tokens, output_tokens=1024)
print(f"Estimated cost: ${cost:.6f}")

# 3. Pengecekan Limit Konteks
CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5":   200_000,
    "gpt-4o":            128_000,
    "gpt-4o-mini":       128_000,
    "gemini-1.5-pro":    1_000_000,
    "qwen/qwen3.8-max:free": 32_000, # Perkiraan batas konteks ditambahkan
}

def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit

print(f"Fits in context: {fits_in_context('qwen/qwen3.8-max:free', response_tokens.input_tokens)}")

# 4. Mengeksekusi permintaan API yang sebenarnya
message = client.messages.create(
    model="qwen/qwen3.8-max:free",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)

print("\n--- Model Response ---")
print(message.content[0].text)
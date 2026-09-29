import os
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv

import anthropic
from anthropic import Anthropic, AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import OpenAI, AsyncOpenAI, RateLimitError as OpenAIRateLimitError

load_dotenv()

# ==========================================
# 1. Definisi Fungsi & Class
# ==========================================

def retry_on_rate_limit(client, model: str, messages: list, max_retries: int = 5):
    """Catches rate limit errors and retries with exponential backoff."""
    for attempt in range(max_retries):
        try:
            if isinstance(client, OpenAI):
                return client.chat.completions.create(
                    model=model,
                    messages=messages
                )
            elif isinstance(client, Anthropic):
                return client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=messages
                )
            else:
                raise ValueError("Unsupported client type.")
                
        except (OpenAIRateLimitError, AnthropicRateLimitError) as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = 2 ** attempt
            print(f"Rate limited. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)


class BudgetExceeded(Exception):
    """Custom exception raised when token budget is exceeded."""
    pass

class TokenBudgetManager:
    """Tracks cumulative token usage across multiple calls."""
    def __init__(self, limit: int):
        self.limit = limit
        self.cumulative_usage = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.cumulative_usage += (input_tokens + output_tokens)
        if self.cumulative_usage > self.limit:
            raise BudgetExceeded(f"Budget exceeded! Used {self.cumulative_usage} / {self.limit} tokens.")
        return self.cumulative_usage


async def _fetch_openai(prompt: str, model: str) -> dict:
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
    )
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.time() - start) * 1000
    return {
        "model": model,
        "response_text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": latency_ms
    }

async def _fetch_anthropic(prompt: str, model: str) -> dict:
    client = AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="https://api.xkiro.com",
    )
    start = time.time()
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.time() - start) * 1000
    return {
        "model": model,
        "response_text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms
    }

async def _compare_models_async(prompt: str, models: list[str]) -> pd.DataFrame:
    tasks = []
    for model in models:
        if model.startswith("openai/"):
            tasks.append(_fetch_openai(prompt, model))
        else:
            tasks.append(_fetch_anthropic(prompt, model))
            
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
    """Calls the same prompt on multiple models concurrently and returns a DataFrame."""
    return asyncio.run(_compare_models_async(prompt, models))


def stream_to_file(prompt: str, output_path: str):
    """Uses the Anthropic streaming API to write tokens to a file in real time."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="https://api.xkiro.com",
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with client.messages.stream(
            model="qwen/qwen3.8-max:free",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                f.write(text)
                f.flush()


# ==========================================
# 2. Blok Eksekusi (Untuk Mencetak Hasil)
# ==========================================
if __name__ == "__main__":
    print("Memulai pengujian...\n")
    
    # Inisialisasi klien untuk tes
    client_openai = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
    )
    
    test_prompt = "What is retrieval-augmented generation in one short sentence?"

    # --- Latihan 1 ---
    print("--- 1. Testing retry_on_rate_limit ---")
    res = retry_on_rate_limit(client_openai, "openai/gpt-4o-mini", [{"role": "user", "content": test_prompt}])
    print(f"Respons OpenAI: {res.choices[0].message.content}\n")

    # --- Latihan 2 ---
    print("--- 2. Testing TokenBudgetManager ---")
    manager = TokenBudgetManager(limit=1000)
    print(f"Limit Token: {manager.limit}")
    current = manager.add_usage(input_tokens=150, output_tokens=300)
    print(f"Total penggunaan setelah API call ke-1: {current}")
    try:
        print("Simulasi API call ke-2 yang melebihi limit...")
        manager.add_usage(input_tokens=200, output_tokens=500) 
    except BudgetExceeded as e:
        print(f"Error berhasil ditangkap: {e}\n")

    # --- Latihan 3 ---
    print("--- 3. Testing compare_models ---")
    df = compare_models(test_prompt, ["openai/gpt-4o-mini", "qwen/qwen3.8-max:free"])
    print(df.to_string())
    print("\n")

    # --- Latihan 4 ---
    print("--- 4. Testing stream_to_file ---")
    filename = "streaming_output.txt"
    stream_to_file(test_prompt, filename)
    print(f"Selesai! Isi file '{filename}' adalah:")
    with open(filename, 'r', encoding='utf-8') as f:
        print(f.read())
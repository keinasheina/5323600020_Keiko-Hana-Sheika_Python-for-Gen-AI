from abc import ABC, abstractmethod
from dataclasses import dataclass
import anthropic, os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Memanggil find_dotenv() agar bisa mencari file .env di luar folder modul
load_dotenv(find_dotenv())

@dataclass
class ChatMessage:
    role: str      # "user" or "assistant"
    content: str

@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse: ...

class AnthropicClient(BaseLLMClient):
    # Model default disesuaikan ke Qwen
    def __init__(self, model: str = "qwen/qwen3.8-max:free"):
        self.model = model
        # Client disesuaikan dengan konfigurasi Xkiro
        self._client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.xkiro.com",
        )

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

class OpenAIClient(BaseLLMClient):
    # Model default disesuaikan ke gpt-4o-mini via OpenRouter
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
        # Client disesuaikan dengan konfigurasi OpenRouter
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/",
                "X-Title": "Python For GenAI",
            },
        )

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )


# --- BAGIAN PENGGUNAAN (USAGE) ---
# Pesan yang sama akan dikirim ke dua backend yang berbeda secara berurutan

msgs = [ChatMessage(role="user", content="What is retrieval-augmented generation?")]

print("=== Menjalankan Qwen via Xkiro (Anthropic SDK) ===")
client_qwen: BaseLLMClient = AnthropicClient()
result_qwen = client_qwen.chat(msgs, system="Be concise.")
print(result_qwen.text)
print(f"Cost estimate: {result_qwen.input_tokens} in, {result_qwen.output_tokens} out\n")


print("=== Menjalankan GPT-4o-Mini via OpenRouter (OpenAI SDK) ===")
client_gpt: BaseLLMClient = OpenAIClient()
result_gpt = client_gpt.chat(msgs, system="Be concise.")
print(result_gpt.text)
print(f"Cost estimate: {result_gpt.input_tokens} in, {result_gpt.output_tokens} out")
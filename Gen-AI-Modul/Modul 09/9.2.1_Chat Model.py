from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI model via OpenRouter
llm_oai = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# Anthropic model via custom proxy
llm_ant = ChatAnthropic(
    model="qwen/qwen3.8-max:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.xkiro.com",
)

# Both accept the same message types
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

# Invoke (blocking)
response = llm_ant.invoke(messages)
print(response.content)

# Stream
for chunk in llm_ant.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# Batch - run multiple inputs in parallel
responses = llm_ant.batch([
    [HumanMessage(content="What is RAG?")],
    [HumanMessage(content="What is a vector database?")],
])
for r in responses:
    print(r.content[:80])
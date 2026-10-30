import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Inisialisasi LLM via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# Pydantic Schemas untuk Structured Output
class Facts(BaseModel):
    facts: List[str] = Field(description="List of exactly 3 key facts extracted from the document")

class Quiz(BaseModel):
    questions: List[str] = Field(description="List of quiz questions based on the key facts")

# 1. Prompt Ekstraksi Fakta
facts_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract exactly 3 key facts from the following text."),
    ("human", "{text}"),
])

# 2. Prompt Pembuatan Kuis
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 1 quiz question for each of the following key facts."),
    ("human", "{facts}"),
])

# Menyusun LCEL Chain
fact_quiz_chain = (
    facts_prompt
    | llm.with_structured_output(Facts)
    | (lambda facts_obj: {"facts": "\n".join(f"- {f}" for f in facts_obj.facts)})
    | quiz_prompt
    | llm.with_structured_output(Quiz)
    | (lambda quiz_obj: quiz_obj.questions)
)

# Pengujian Script
raw_document = """
LangChain is an open-source framework designed to simplify the creation of applications using LLMs.
It provides modular abstractions for models, prompts, and memory, allowing developers to chain components together.
LangChain Expression Language (LCEL) offers a declarative way to build complex, production-ready pipelines with streaming and async support.
"""

questions_list = fact_quiz_chain.invoke({"text": raw_document})

print("Hasil Python List:")
print(questions_list)
print("\nTipe data output:", type(questions_list))
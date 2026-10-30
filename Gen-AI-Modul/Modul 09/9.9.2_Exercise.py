import hashlib
import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

# Setup Embeddings & Vectorstore
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

docs = [
    Document(page_content="RAG combines retrieval systems with generative LLMs to answer questions."),
    Document(page_content="Vector databases like Chroma store document embeddings for fast similarity search."),
    Document(page_content="Fine-tuning modifies model weights directly, while RAG feeds context dynamically."),
    Document(page_content="Retrieval-Augmented Generation helps LLMs avoid hallucinations on unseen data."),
]

vectorstore = Chroma.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Setup LLM & Prompt Paraphrase
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

class ParaphrasedQueries(BaseModel):
    queries: List[str] = Field(description="3 paraphrased variations of the user query")

paraphrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 different paraphrased variations of the user query to improve retrieval."),
    ("human", "{query}"),
])

query_generator = paraphrase_prompt | llm.with_structured_output(ParaphrasedQueries)

def manual_multi_query_retriever(query: str) -> List[Document]:
    # 1. Generate 3 paraphrased queries
    paraphrases = query_generator.invoke({"query": query}).queries
    all_queries = [query] + paraphrases
    
    print("Variasi Query yang digunakan:")
    for idx, q in enumerate(all_queries, 1):
        print(f" {idx}. {q}")
    print()

    seen_hashes = set()
    unique_documents = []

    # 2. Retrieve & 3. Deduplicate by content hash
    for q in all_queries:
        retrieved_docs = base_retriever.invoke(q)
        for doc in retrieved_docs:
            content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)

    return unique_documents

# Pengujian Script
user_query = "How does RAG prevent hallucinations in language models?"
results = manual_multi_query_retriever(user_query)

print(f"Total unik dokumen terambil: {len(results)}\n")
for i, doc in enumerate(results, 1):
    print(f"Dokumen {i}: {doc.page_content}")
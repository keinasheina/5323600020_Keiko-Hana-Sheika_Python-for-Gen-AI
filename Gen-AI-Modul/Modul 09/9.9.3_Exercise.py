import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

parser = StrOutputParser()

# 1. Map Step (Merangkum per chunk)
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text chunk concisely:"),
    ("human", "{chunk}"),
])
map_chain = map_prompt | llm | parser

# 2. Reduce Step (Gabungkan seluruh rangkuman)
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "Synthesize the following list of chunk summaries into a single cohesive final summary:"),
    ("human", "{summaries}"),
])
reduce_chain = reduce_prompt | llm | parser

def map_reduce_summarize(long_text: str) -> str:
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(long_text)
    print(f"Dokumen dibagi menjadi {len(chunks)} chunks.")

    # MAP: Rangkum masing-masing chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        summary = map_chain.invoke({"chunk": chunk})
        chunk_summaries.append(summary)
        print(f"Summary Chunk {i}: {summary[:80]}...")

    # REDUCE: Gabungkan rangkuman
    combined_summaries = "\n".join(f"- {s}" for s in chunk_summaries)
    final_summary = reduce_chain.invoke({"summaries": combined_summaries})
    
    return final_summary

# Pengujian Script
long_document = """
Large Language Models (LLMs) have transformed natural language processing across various domains. 
However, they face challenges such as hallucinations and outdated knowledge when answering niche queries.
To overcome this, Retrieval-Augmented Generation (RAG) was introduced to inject external context at query time.
By indexing documents into a vector database, systems can perform fast vector similarity searches to retrieve accurate knowledge.
When combined with agentic workflows and memory buffers, LLMs can execute multi-step calculations and maintain context across long conversations.
Finally, evaluation techniques like groundedness scoring ensure that outputs adhere strictly to provided context without inventing facts.
"""

print("--- Hasil Akhir Map-Reduce Summary ---")
result = map_reduce_summarize(long_document)
print("\n" + result)
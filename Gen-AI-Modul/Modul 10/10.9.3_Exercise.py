# 10.9_Exercise_3.py
import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Setup LLM dan Embedding
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

# Corpus dengan metadata yang ditandai
documents = [
    Document(text="Artikel tentang dasar-dasar Word2Vec.", metadata={"year": 2018, "topic": "embeddings"}),
    Document(text="Panduan text-embedding-3 dari OpenAI.", metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Cara fine-tuning model LLM Llama-3.", metadata={"year": 2024, "topic": "fine-tuning"}),
    Document(text="Metode approximate nearest-neighbor HNSW.", metadata={"year": 2024, "topic": "embeddings"}),
]

index = VectorStoreIndex.from_documents(documents)

def metadata_filtered_search(query: str, target_year: int, target_topic: str):
    # Menerapkan filter gabungan
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="year", value=target_year),
            ExactMatchFilter(key="topic", value=target_topic),
        ]
    )
    
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)
    retrieved_nodes = retriever.retrieve(query)
    
    print(f"Hasil pencarian untuk: '{query}' (Filter: year={target_year}, topic={target_topic})")
    for n in retrieved_nodes:
        print(f"- [Score: {n.score:.4f}] {n.text} | Metadata: {n.metadata}")

if __name__ == "__main__":
    metadata_filtered_search("teknologi vektor", 2024, "embeddings")
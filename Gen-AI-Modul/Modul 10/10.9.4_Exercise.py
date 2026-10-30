# 10.9_Exercise_4.py
import os
import re
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

llm_client = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

Settings.llm = llm_client
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

# 12 Kalimat dokumen sebagai data source
data_source = [
    "JavaScript digunakan secara luas di pengembangan web.",
    "Python adalah bahasa utama untuk analisis data dan machine learning.",
    "LlamaIndex memudahkan pembuatan aplikasi LLM dengan RAG.",
    "React adalah library frontend yang dibuat oleh Facebook.",
    "RAG meningkatkan performa LLM tanpa perlu fine-tuning.",
    "Docker digunakan untuk manajemen kontainer aplikasi.",
    "Vector Database menyimpan embedding untuk semantic search.",
    "Pandas dan NumPy adalah library Python populer.",
    "Prompt engineering membantu menghasilkan output LLM yang lebih presisi.",
    "Node.js memungkinkan JavaScript berjalan di sisi server.",
    "OpenAI merilis GPT-4 dengan kapabilitas multimodal.",
    "ChromaDB adalah basis data vektor open-source populer."
]
documents = [Document(text=text) for text in data_source]
index = VectorStoreIndex.from_documents(documents)

def score_relevance(question: str, node_text: str) -> float:
    """Memanggil LLM untuk memberi skor relevansi (0-10)"""
    prompt = f"""Nilai relevansi dokumen berikut terhadap pertanyaan pengguna dari skala 0 hingga 10.
Berikan HANYA angkanya saja (misal: 8 atau 9.5). Jangan tambahkan kata lain.

Pertanyaan: {question}
Dokumen: {node_text}
Skor:"""
    
    response = llm_client.complete(prompt)
    match = re.search(r"(\d+(?:\.\d+)?)", str(response))
    return float(match.group(1)) if match else 0.0

def two_stage_retriever(query: str):
    # Tahap 1: Ambil top-10 menggunakan VectorIndexRetriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    candidate_nodes = retriever.retrieve(query)
    print(f"Tahap 1: Mengambil {len(candidate_nodes)} node awal.\n")
    
    # Tahap 2: LLM Scoring & Re-ranking
    scored_nodes = []
    for node in candidate_nodes:
        score = score_relevance(query, node.text)
        scored_nodes.append((score, node))
        print(f"  [Tahap 2 Scoring] Teks: '{node.text}' -> Skor: {score}")
        
    # Urutkan secara menurun berdasarkan skor LLM
    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    
    # Kembalikan top-3
    top_3 = scored_nodes[:3]
    
    print("\n--- HASIL AKHIR (TOP 3) ---")
    for rank, (score, node) in enumerate(top_3, 1):
        print(f"{rank}. [Skor LLM: {score}/10] {node.text}")

if __name__ == "__main__":
    two_stage_retriever("Apa itu RAG dan teknologi di baliknya?")
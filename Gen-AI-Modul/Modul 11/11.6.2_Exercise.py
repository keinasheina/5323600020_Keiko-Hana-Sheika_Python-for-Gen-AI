import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

class MultiCollectionSearch:
    def __init__(self, client: chromadb.Client):
        self.client = client

    def search(self, query: str, collection_names: list[str], top_k: int = 5) -> list[dict]:
        q_emb = embed([query])[0]
        combined_results = []

        for name in collection_names:
            try:
                coll = self.client.get_collection(name)
                res = coll.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"]
                )
                
                docs = res["documents"][0]
                dists = res["distances"][0]
                metas = res["metadatas"][0]

                for doc, dist, meta in zip(docs, dists, metas):
                    # ChromaDB Cosine distance: 0 = identik, convert ke similarity score [0, 1]
                    similarity = 1.0 - (dist / 2.0)
                    combined_results.append({
                        "collection": name,
                        "document": doc,
                        "score": round(similarity, 4),
                        "metadata": meta
                    })
            except Exception as e:
                print(f"Peringatan: Gagal melakukan query pada koleksi {name}: {e}")

        # Urutkan berdasarkan score tertinggi
        sorted_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

# Testing Script
if __name__ == "__main__":
    client = chromadb.Client()
    
    # Buat 2 koleksi dummy
    c1 = client.get_or_create_collection("tech_docs", metadata={"hnsw:space": "cosine"})
    c2 = client.get_or_create_collection("ai_docs", metadata={"hnsw:space": "cosine"})

    t1 = ["Python is a widely used programming language.", "Docker helps in containerization."]
    t2 = ["RAG retrieves external knowledge for LLMs.", "Embeddings convert text into vectors."]

    c1.add(ids=["t1", "t2"], documents=t1, embeddings=embed(t1), metadatas=[{"src": "tech"}, {"src": "tech"}])
    c2.add(ids=["a1", "a2"], documents=t2, embeddings=embed(t2), metadatas=[{"src": "ai"}, {"src": "ai"}])

    multi_search = MultiCollectionSearch(client)
    results = multi_search.search("How to add knowledge to LLM?", ["tech_docs", "ai_docs"], top_k=3)

    print("=== HASIL SEARCH DARI MULTI-COLLECTION ===")
    for r in results:
        print(f"[{r['score']}] ({r['collection']}) {r['document']}")
import os
import time
import numpy as np
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Client OpenRouter
openai_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

def get_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return resp.data[0].embedding

class VectorDBBenchmark:
    def __init__(self, collection_name: str = "benchmark_coll"):
        self.client = chromadb.Client() # In-memory client
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def prepare_data(self, num_docs: int = 1000):
        print(f"Menyiapkan {num_docs} dokumen mock...")
        # Menggunakan mock embedding vektor acak (1536 dim) untuk mempercepat pengisian data
        rng = np.random.default_rng(42)
        dim = 1536
        
        documents = [f"Dokumen mock nomor {i} untuk benchmark." for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        # Normalisasi vektor agar kompatibel dengan cosine similarity
        raw_embeddings = rng.standard_normal((num_docs, dim))
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        embeddings = (raw_embeddings / norms).tolist()

        # Insert dalam batch
        batch_size = 200
        for i in range(0, num_docs, batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size]
            )
        print("Data berhasil dimasukkan ke ChromaDB.")

    def run_benchmark(self, num_queries: int = 100, k: int = 5):
        print(f"Menjalankan {num_queries} query benchmark...")
        latencies = []
        
        # Vektor query acak untuk benchmarking
        rng = np.random.default_rng(123)
        raw_queries = rng.standard_normal((num_queries, 1536))
        norms = np.linalg.norm(raw_queries, axis=1, keepdims=True)
        query_embeddings = (raw_queries / norms).tolist()

        start_total = time.perf_counter()
        
        for q_emb in query_embeddings:
            q_start = time.perf_counter()
            self.collection.query(
                query_embeddings=[q_emb],
                n_results=k
            )
            q_end = time.perf_counter()
            latencies.append((q_end - q_start) * 1000) # ms

        end_total = time.perf_counter()
        total_time_sec = end_total - start_total

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        throughput = num_queries / total_time_sec

        print("\n=== HASIL BENCHMARK ===")
        print(f"Total Query        : {num_queries}")
        print(f"Average Latency    : {avg_latency:.2f} ms")
        print(f"P95 Latency        : {p95_latency:.2f} ms")
        print(f"Throughput         : {throughput:.2f} queries/sec")

if __name__ == "__main__":
    benchmark = VectorDBBenchmark()
    benchmark.prepare_data(num_docs=1000)
    benchmark.run_benchmark(num_queries=100)
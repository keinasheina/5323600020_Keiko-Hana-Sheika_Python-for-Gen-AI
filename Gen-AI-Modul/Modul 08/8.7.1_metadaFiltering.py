import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables & Inisialisasi Klien OpenRouter
load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# 2. Definisi Struktur Dokumen & Fungsi Embedding
@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

def embed_texts(texts: list[str]) -> np.ndarray:
    """Membuat embedding teks menggunakan model dari OpenRouter/OpenAI."""
    resp = client.embeddings.create(
        input=texts, 
        model="openai/text-embedding-3-small"
    )
    vecs = np.array([e.embedding for e in sorted(resp.data, key=lambda x: x.index)], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)

# 3. Implementasi FilteredVectorStore
class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        embeddings = embed_texts([d.text for d in docs])
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Pencarian dengan filter metadata opsional sebelum perangkingan."""
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        q_vec = embed_texts([query])[0]
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(candidates[i], float(scores[i])) for i in top_idx]

# 4. Struktur Data & Fungsi Evaluasi Retrieval
@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(
    store: FilteredVectorStore,
    eval_cases: list[RetrievalEvalCase],
    k: int = 5,
) -> dict:
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        # Mengambil ID dokumen dari hasil tuple (document, score)
        retrieved_ids = [r[0].id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}":    round(float(np.mean(r_scores)), 4),
        "MRR":            round(float(np.mean(mrr_scores)), 4),
    }

# 5. Inisialisasi Korpus Data dan Menjalankan Evaluasi
docs = [
    FilteredDocument("d01", "Retrieval-Augmented Generation combines search with language models.", {"category": "rag", "year": 2024}),
    FilteredDocument("d02", "Vector databases store embeddings for efficient similarity search.", {"category": "db", "year": 2024}),
    FilteredDocument("d03", "Fine-tuning adapts a pre-trained model to specific tasks.", {"category": "training", "year": 2024}),
    FilteredDocument("d08", "RAG systems retrieve relevant context before generating answers.", {"category": "rag", "year": 2024}),
    FilteredDocument("d09", "Transformers use self-attention mechanisms to relate tokens.", {"category": "nlp", "year": 2023}),
    FilteredDocument("d10", "Autonomous agents plan and execute tasks using LLMs.", {"category": "agents", "year": 2024}),
]

fstore = FilteredVectorStore()
fstore.add(docs)

eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
]

metrics = evaluate_retrieval(fstore, eval_cases, k=3)
print(metrics)
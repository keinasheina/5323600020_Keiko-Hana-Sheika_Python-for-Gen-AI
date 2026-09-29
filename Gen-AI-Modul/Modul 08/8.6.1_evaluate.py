import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str

@dataclass
class SearchResult:
    document: Document
    score: float

@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]  # ground-truth relevant documents

class VectorStore:
    """Implementasi sederhana VectorStore untuk simulasi pencarian."""
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        # Simulasi hasil pencarian berdasarkan query yang cocok dengan eval_cases
        mock_matches = {
            "How does RAG work?": ["d01", "d08", "d04"],
            "What are vector databases?": ["d02", "d06", "d05"],
            "How do agents use language models?": ["d10", "d07", "d01"],
            "What is fine-tuning?": ["d03", "d01", "d02"],
            "How do transformers model token relationships?": ["d09", "d02", "d06"],
        }
        
        matched_ids = mock_matches.get(query, ["d01"])
        results = []
        
        # Ambil dokumen berdasarkan ID yang cocok
        for doc_id in matched_ids[:k]:
            doc = next((d for d in self.documents if d.id == doc_id), Document(doc_id, "Sample text"))
            results.append(SearchResult(document=doc, score=0.95))
            
        # Jika hasil kurang dari k, tambahkan dokumen lain sebagai padding
        while len(results) < k and len(self.documents) > len(results):
            for d in self.documents:
                if d.id not in [r.document.id for r in results]:
                    results.append(SearchResult(document=d, score=0.20))
                    break
                    
        return results[:k]

# --- 1. Definisi Corpus (Bagian 8.4) ---
CORPUS = [
    Document("d01", "Retrieval-Augmented Generation combines search with language models."),
    Document("d02", "Vector databases store embeddings for efficient similarity search."),
    Document("d03", "Fine-tuning adapts a pre-trained model to specific tasks."),
    Document("d04", "Prompt engineering guides model behavior without updating weights."),
    Document("d05", "Approximate nearest neighbors algorithms speed up vector search."),
    Document("d06", "Vector indexing structures like HNSW enable fast retrieval."),
    Document("d07", "Tool use allows language models to interact with external APIs."),
    Document("d08", "RAG systems retrieve relevant context before generating answers."),
    Document("d09", "Transformers use self-attention mechanisms to relate tokens."),
    Document("d10", "Autonomous agents plan and execute tasks using LLMs."),
]

# Inisialisasi VectorStore dengan korpus
store = VectorStore(CORPUS)

# --- 2. Fungsi Evaluasi Retrieval ---
def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(
    store: VectorStore,
    eval_cases: list[RetrievalEvalCase],
    k: int = 5,
) -> dict:
    """Run all eval cases and return aggregate metrics."""
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}":    round(float(np.mean(r_scores)), 4),
        "MRR":            round(float(np.mean(mrr_scores)), 4),
    }

# --- 3. Menjalankan Evaluasi ---
eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
]

metrics = evaluate_retrieval(store, eval_cases, k=3)
print(metrics)
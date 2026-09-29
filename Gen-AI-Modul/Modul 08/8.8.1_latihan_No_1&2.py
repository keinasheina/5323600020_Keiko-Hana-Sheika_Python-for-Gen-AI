import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional  # <--- Tambahkan Optional di sini
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Konfigurasi Klien OpenAI via OpenRouter
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

def embed_texts(texts: list[str], model: str = "openai/text-embedding-3-small") -> np.ndarray:
    """Helper untuk membuat embedding menggunakan OpenRouter."""
    resp = client.embeddings.create(input=texts, model=model)
    vecs = np.array([e.embedding for e in sorted(resp.data, key=lambda x: x.index)], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


# ==========================================
# 1. Implementasi DuplicateDetector
# ==========================================
class DuplicateDetector:
    def __init__(self, documents: List[Document], threshold: float = 0.95):
        self.documents = documents
        self.threshold = threshold

    def find_duplicates(self) -> List[Tuple[Document, Document, float]]:
        texts = [d.text for d in self.documents]
        embeddings = embed_texts(texts)
        
        # Hitung matrix cosine similarity
        similarity_matrix = embeddings @ embeddings.T
        duplicates = []

        n = len(self.documents)
        for i in range(n):
            for j in range(i + 1, n):
                score = similarity_matrix[i, j]
                if score >= self.threshold:
                    duplicates.append((self.documents[i], self.documents[j], float(score)))

        return duplicates


# ==========================================
# 2. Implementasi HybridSearch
# ==========================================
class HybridSearch:
    def __init__(self, documents: List[Document], alpha: float = 0.5):
        self.documents = documents
        self.alpha = alpha
        self.embeddings = embed_texts([d.text for d in documents])

    def _keyword_score(self, query: str, doc_text: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())
        if not query_terms:
            return 0.0
        intersection = query_terms.intersection(doc_terms)
        return len(intersection) / len(query_terms)

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []

        # Hitung skor semantik
        q_vec = embed_texts([query])[0]
        semantic_scores = self.embeddings @ q_vec

        # Hitung skor keyword
        keyword_scores = np.array([self._keyword_score(query, d.text) for d in self.documents])

        # Gabungkan skor menggunakan parameter alpha
        final_scores = (self.alpha * semantic_scores) + ((1 - self.alpha) * keyword_scores)

        k = min(k, len(self.documents))
        top_idx = np.argsort(final_scores)[::-1][:k]
        return [(self.documents[i], float(final_scores[i])) for i in top_idx]


# --- Contoh Pengujian Bagian 1 ---
if __name__ == "__main__":
    corpus = [
        Document("d01", "Retrieval-Augmented Generation combines search with language models."),
        Document("d02", "Retrieval-Augmented Generation combines search with language models (duplicate)."),
        Document("d03", "Vector databases store embeddings for efficient similarity search."),
    ]

    print("--- Uji DuplicateDetector ---")
    detector = DuplicateDetector(corpus, threshold=0.90)
    print(detector.find_duplicates())

    print("\n--- Uji HybridSearch ---")
    hybrid = HybridSearch(corpus, alpha=0.7)
    print(hybrid.search("vector databases", k=2))
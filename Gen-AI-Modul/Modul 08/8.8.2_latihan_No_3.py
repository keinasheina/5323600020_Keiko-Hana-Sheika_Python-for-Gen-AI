import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi client menggunakan OpenRouter
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# ── Data structures ────────────────────────────────────────
@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

# ── Embedding helper ───────────────────────────────────────
def embed_batch(texts: list[str], model: str = "openai/text-embedding-3-small") -> np.ndarray:
    """Embed texts in a single API call. Returns (n, dim) float32 array."""
    response = client.embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

# ── Simple in-memory vector store ──────────────────────────
class VectorStore:
    """
    In-memory vector store for semantic search with support for add, search, delete, and update.
    """
    def __init__(self, embed_model: str = "openai/text-embedding-3-small"):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) normalised

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index a list of documents."""
        texts = [d.text for d in documents]
        vectors = embed_batch(texts, model=self.embed_model)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)
        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)
        
        self._matrix = np.array(
            [d.embedding for d in self._documents], dtype=np.float32
        )
        print(f"Index now contains {len(self._documents)} documents.")

    def delete(self, doc_id: str) -> bool:
        """Delete a document by its ID and maintain matrix consistency."""
        initial_len = len(self._documents)
        self._documents = [d for d in self._documents if d.id != doc_id]
        
        if len(self._documents) == initial_len:
            print(f"Document with ID {doc_id!r} not found.")
            return False
        
        if len(self._documents) == 0:
            self._matrix = None
        else:
            self._matrix = np.array(
                [d.embedding for d in self._documents], dtype=np.float32
            )
        
        print(f"Document {doc_id!r} deleted. Index now contains {len(self._documents)} documents.")
        return True

    def update(self, doc_id: str, new_text: str) -> bool:
        """Update a document's text, re-embed it, and maintain matrix consistency."""
        doc_to_update = None
        for doc in self._documents:
            if doc.id == doc_id:
                doc_to_update = doc
                break
        
        if doc_to_update is None:
            print(f"Document with ID {doc_id!r} not found for update.")
            return False
        
        # Perbarui teks dokumen dan buat embedding baru yang ternormalisasi
        doc_to_update.text = new_text
        vectors = embed_batch([new_text], model=self.embed_model)
        vec = vectors[0]
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = (vec / norm).astype(np.float32)
        else:
            vec = vec.astype(np.float32)
        
        doc_to_update.embedding = vec
        
        # Bangun ulang matriks agar konsisten dengan embedding baru
        self._matrix = np.array(
            [d.embedding for d in self._documents], dtype=np.float32
        )
        print(f"Document {doc_id!r} updated and re-embedded.")
        return True

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the k most similar documents for a query string."""
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("No documents indexed yet.")
        
        q_vec = embed_batch([query], model=self.embed_model)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = (q_vec / q_norm).astype(np.float32)
        
        scores = self._matrix @ q_vec
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

    @property
    def size(self) -> int:
        return len(self._documents)

# ── Demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    CORPUS = [
        Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation."),
        Document("d02", "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search."),
    ]

    store = VectorStore()
    store.add_documents(CORPUS)

    print("\n--- Testing Delete ---")
    store.delete("d01")

    print("\n--- Testing Update ---")
    store.update("d02", "Pinecone, Milvus, and Chroma are popular vector databases for AI applications.")

    print("\n--- Testing Search After Updates ---")
    results = store.search("vector database", k=2)
    for r in results:
        print(f"  [{r.rank}] score={r.score:.4f} | {r.document.text}")
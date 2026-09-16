import numpy as np


# ==========================================
# Corpus
# ==========================================

corpus = [
    "Artificial intelligence is transforming technology.",
    "Artificial intelligence is changing modern technology.",
    "Cats are common household animals.",
    "Dogs are popular household pets.",
    "Machine learning is a part of artificial intelligence."
]


# ==========================================
# Hash-based mock embedding
# ==========================================

def mock_embedding(
    text: str,
    dimension: int = 128
) -> np.ndarray:
    """
    Membuat embedding sederhana menggunakan hash.
    """

    seed = abs(hash(text)) % (2**32)

    rng = np.random.default_rng(seed)

    return rng.normal(
        size=dimension
    )


# ==========================================
# Membuat embedding
# ==========================================

embeddings = np.array([
    mock_embedding(text)
    for text in corpus
])


# ==========================================
# Normalisasi embedding
# ==========================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

embeddings = embeddings / norms


# ==========================================
# Cosine similarity matrix
# ==========================================

similarity_matrix = embeddings @ embeddings.T


print("=== COSINE SIMILARITY MATRIX ===")
print(
    np.round(similarity_matrix, 4)
)


# ==========================================
# Cari pasangan dengan similarity tertinggi
# ==========================================

highest_similarity = -1
best_pair = None


for i in range(len(corpus)):

    for j in range(i + 1, len(corpus)):

        similarity = similarity_matrix[i, j]

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_pair = (i, j)


# ==========================================
# Print hasil
# ==========================================

i, j = best_pair

print("\n=== PAIR DENGAN SIMILARITY TERTINGGI ===")

print(f"String 1: {corpus[i]}")
print(f"String 2: {corpus[j]}")
print(f"Similarity: {highest_similarity:.4f}")
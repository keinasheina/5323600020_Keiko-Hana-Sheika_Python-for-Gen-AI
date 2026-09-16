import numpy as np


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise setiap baris pada matrix.
    """

    matrix = np.asarray(matrix, dtype=float)

    # Hitung L2 norm setiap baris
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    # Hindari pembagian dengan nol
    norms = np.where(norms == 0, 1, norms)

    # Normalisasi
    return matrix / norms


# ==========================================
# Contoh embedding
# ==========================================

matrix = np.array([
    [3, 4, 0],
    [1, 2, 2],
    [5, 0, 12],
    [2, 3, 6]
], dtype=float)


print("=== MATRIX AWAL ===")
print(matrix)


# ==========================================
# Normalisasi
# ==========================================

normalised = normalise_embeddings(matrix)

print("\n=== MATRIX SETELAH NORMALISASI ===")
print(np.round(normalised, 4))


# ==========================================
# Verifikasi row norms
# ==========================================

row_norms = np.linalg.norm(
    normalised,
    axis=1
)

print("\n=== ROW NORMS ===")
print(np.round(row_norms, 6))


# Cek apakah semua norm = 1
all_norm_one = np.allclose(
    row_norms,
    1.0
)

print(
    "\nSemua row norm = 1.0:",
    all_norm_one
)

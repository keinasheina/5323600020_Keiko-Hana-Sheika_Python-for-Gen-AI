import pandas as pd
from pathlib import Path
import re


def analyse_txt_folder(folder_path: str | Path) -> pd.DataFrame:
    """
    Membaca semua file .txt dalam folder.

    Menghasilkan DataFrame dengan kolom:
    filename
    char_count
    word_count
    sentence_count

    Data diurutkan berdasarkan word_count
    dari terbesar ke terkecil.
    """

    folder = Path(folder_path)

    results = []

    # Membaca semua file .txt
    for file_path in folder.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        # Jumlah karakter
        char_count = len(text)

        # Jumlah kata
        words = re.findall(
            r"\b\w+\b",
            text
        )

        word_count = len(words)

        # Jumlah kalimat
        sentences = re.findall(
            r"[^.!?]+[.!?]+",
            text
        )

        sentence_count = len(sentences)

        results.append({
            "filename": file_path.name,
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count
        })

    # Membuat DataFrame
    df = pd.DataFrame(results)

    # Sort word_count descending
    df = df.sort_values(
        by="word_count",
        ascending=False
    ).reset_index(drop=True)

    return df


# ==========================================
# Folder TXT
# ==========================================

BASE_DIR = Path(__file__).parent
txt_folder = BASE_DIR / "txt_files"


# ==========================================
# Jalankan fungsi
# ==========================================

result = analyse_txt_folder(txt_folder)


print("=== HASIL ANALISIS FILE TXT ===")
print(result.to_string(index=False))
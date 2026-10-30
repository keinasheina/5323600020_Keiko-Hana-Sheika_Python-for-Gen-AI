import os
import json
import numpy as np
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

def export_collection_to_json(collection, output_filepath: str):
    """Mengekspor isi koleksi ChromaDB ke file JSON."""
    count = collection.count()
    if count == 0:
        print("Koleksi kosong. Tidak ada yang diekspor.")
        return

    data = collection.get(include=["documents", "embeddings", "metadatas"])
    
    export_list = []
    for i in range(len(data["ids"])):
        # Konversi embedding ke list jika berupa numpy ndarray
        embedding_val = data["embeddings"][i]
        if isinstance(embedding_val, np.ndarray):
            embedding_val = embedding_val.tolist()
            
        item = {
            "id": data["ids"][i],
            "text": data["documents"][i],
            "embedding": embedding_val,
            "metadata": data["metadatas"][i] if data["metadatas"] else {}
        }
        export_list.append(item)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(export_list, f, indent=2)
    
    print(f"Berhasil mengekspor {len(export_list)} item ke '{output_filepath}'.")

def import_json_to_collection(json_filepath: str, collection):
    """Mengimpor kembali data dari file JSON ke koleksi ChromaDB."""
    if not os.path.exists(json_filepath):
        print(f"File {json_filepath} tidak ditemukan.")
        return

    with open(json_filepath, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if not data_list:
        print("File JSON kosong.")
        return

    ids = [item["id"] for item in data_list]
    documents = [item["text"] for item in data_list]
    embeddings = [item["embedding"] for item in data_list]
    metadatas = [item["metadata"] for item in data_list]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Berhasil mengimpor {len(ids)} item ke dalam koleksi '{collection.name}'.")

# Testing Script
if __name__ == "__main__":
    client = chromadb.Client()

    # 1. Menyiapkan koleksi asal & isi data
    source_coll = client.get_or_create_collection("original_kb", metadata={"hnsw:space": "cosine"})
    sample_texts = ["Vector databases are fast.", "JSON is good for backups."]
    source_coll.add(
        ids=["id1", "id2"],
        documents=sample_texts,
        embeddings=embed(sample_texts),
        metadatas=[{"category": "db"}, {"category": "format"}]
    )

    json_file = "chroma_backup.json"

    # 2. Eksport ke JSON
    export_collection_to_json(source_coll, json_file)

    # 3. Import ke koleksi baru
    target_coll = client.get_or_create_collection("restored_kb", metadata={"hnsw:space": "cosine"})
    import_json_to_collection(json_file, target_coll)

    print(f"Jumlah dokumen koleksi terestorasi: {target_coll.count()}")
    
    # Cleanup file backup temporer
    if os.path.exists(json_file):
        os.remove(json_file)
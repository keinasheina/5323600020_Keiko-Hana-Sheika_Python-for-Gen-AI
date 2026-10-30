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

class TenantVectorStore:
    def __init__(self, persist_dir: str = "/tmp/tenant_chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _get_tenant_collection(self, tenant_id: str):
        # Format nama koleksi berdasarkan tenant_id untuk isolasi
        coll_name = f"tenant_{tenant_id.lower().strip()}"
        return self.client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, tenant_id: str, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        coll = self._get_tenant_collection(tenant_id)
        embeddings = embed(documents)
        coll.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Berhasil menambahkan {len(documents)} dokumen untuk Tenant '{tenant_id}'.")

    def search(self, tenant_id: str, query: str, k: int = 3) -> list[dict]:
        coll = self._get_tenant_collection(tenant_id)
        if coll.count() == 0:
            return []

        q_emb = embed([query])[0]
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=min(k, coll.count()),
            include=["documents", "distances", "metadatas"]
        )

        results = []
        for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
            results.append({
                "document": doc,
                "score": round(1.0 - (dist / 2.0), 4),
                "metadata": meta
            })
        return results

# Testing Script
if __name__ == "__main__":
    store = TenantVectorStore()

    # Tambah dokumen Tenant A
    store.add_documents(
        tenant_id="company_a",
        documents=["Company A financial report 2025.", "Company A internal policy."],
        ids=["a1", "a2"],
        metadatas=[{"confidential": True}, {"confidential": False}]
    )

    # Tambah dokumen Tenant B
    store.add_documents(
        tenant_id="company_b",
        documents=["Company B product roadmap.", "Company B client list."],
        ids=["b1", "b2"],
        metadatas=[{"confidential": True}, {"confidential": True}]
    )

    # Cari data di Tenant A (Data Tenant B tidak akan tercampur)
    print("\n=== PENCARIAN TENANT A ===")
    res_a = store.search(tenant_id="company_a", query="report")
    for r in res_a:
        print(f"[{r['score']}] {r['document']}")

    # Cari data di Tenant B
    print("\n=== PENCARIAN TENANT B ===")
    res_b = store.search(tenant_id="company_b", query="roadmap")
    for r in res_b:
        print(f"[{r['score']}] {r['document']}")
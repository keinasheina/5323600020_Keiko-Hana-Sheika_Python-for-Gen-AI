# 10.9_Exercise_2.py
import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Setup LLM dan Embedding
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

# 10 Paragraf contoh
paragraphs = [
    "1. LlamaIndex pertama kali dikembangkan untuk mempermudah injeksi data kustom ke LLM.",
    "2. VectorStoreIndex memecah teks menjadi node dan menyimpannya dalam basis data vektor.",
    "3. SummaryIndex sangat berguna untuk merangkum seluruh dokumen yang berukuran besar.",
    "4. Pencarian pada basis data vektor menggunakan metrik cosine similarity secara umum.",
    "5. Pengembang merekomendasikan chunk size 512 atau 1024 token untuk hasil yang stabil.",
    "6. Model gpt-4o-mini dari OpenAI menawarkan keseimbangan kecepatan dan efisiensi biaya.",
    "7. Implementasi metadata pada node membantu menyaring hasil pencarian agar lebih presisi.",
    "8. LlamaIndex mendukung berbagai basis data vektor seperti ChromaDB, Pinecone, dan Weaviate.",
    "9. Proses re-ranking dapat diimplementasikan setelah pengambilan node untuk akurasi ekstra.",
    "10. Pada versi terbaru, integrasi LlamaIndex dengan framework aplikasi agen menjadi lebih mulus."
]
documents = [Document(text=p) for p in paragraphs]

# Membuat kedua Index
vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)

# Membuat Query Engine
vector_engine = vector_index.as_query_engine(similarity_top_k=2)
summary_engine = summary_index.as_query_engine()

# 5 Pertanyaan uji (Kombinasi fakta spesifik dan rangkuman)
questions = [
    "Berapa chunk size yang direkomendasikan pengembang?",
    "Basis data vektor apa saja yang didukung oleh LlamaIndex?",
    "Apa fungsi dari VectorStoreIndex?",
    "Jelaskan secara singkat kegunaan SummaryIndex dan VectorStoreIndex berdasarkan teks.",
    "Buat rangkuman keseluruhan poin utama dari teks tersebut."
]

for i, q in enumerate(questions, 1):
    print(f"\n--- Pertanyaan {i}: {q} ---")
    print(f"[VectorStoreIndex] : {vector_engine.query(q)}")
    print(f"[SummaryIndex]     : {summary_engine.query(q)}")
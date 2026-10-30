# 10.9_Exercise_1.py
import os
import tempfile
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Setup LLM dan Embedding menggunakan OpenRouter
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

def setup_dummy_files(folder_path: str):
    """Fungsi pembantu untuk membuat file teks contoh"""
    files = {
        "dokumen_ai.txt": "Kecerdasan Buatan (AI) generatif mampu membuat teks dan gambar baru.",
        "dokumen_rag.txt": "RAG menggabungkan pencarian dokumen dengan pembuatan teks oleh LLM.",
    }
    for nama_file, konten in files.items():
        with open(os.path.join(folder_path, nama_file), "w", encoding="utf-8") as f:
            f.write(konten)

def build_qa_system(folder_path: str):
    # Membaca file .txt dari folder
    reader = SimpleDirectoryReader(input_dir=folder_path, required_exts=[".txt"])
    documents = reader.load_data()
    
    # Mengindeks dokumen
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    def ask(question: str) -> dict:
        response = query_engine.query(question)
        # Mengambil nama file dari metadata node sumber
        source_filenames = list(set(
            node.node.metadata.get("file_name", "Unknown") 
            for node in response.source_nodes
        ))
        
        return {
            "answer": str(response).strip(),
            "source_filenames": source_filenames
        }

    return ask

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        setup_dummy_files(temp_dir)
        
        ask_function = build_qa_system(temp_dir)
        
        # Pengujian fungsi
        pertanyaan = "Apa itu RAG?"
        hasil = ask_function(pertanyaan)
        
        print(f"Pertanyaan: {pertanyaan}")
        print(f"Jawaban: {hasil['answer']}")
        print(f"Sumber File: {hasil['source_filenames']}")
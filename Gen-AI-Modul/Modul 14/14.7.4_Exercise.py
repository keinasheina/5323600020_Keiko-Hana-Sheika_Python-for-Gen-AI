# Nama File: 14.7.4_CompareProcess.py
import os
import time
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# 1. Definisi Agen & Manager
manager = Agent(
    role="Project Manager",
    goal="Mendelegasikan tugas pembuatan ide dan penyuntingan.",
    backstory="Manajer efisien yang mengawasi alur kerja tim.",
    llm=llm,
    allow_delegation=True,
)

ideator = Agent(
    role="Idea Generator",
    goal="Membuat ide cerita singkat tentang AI.",
    backstory="Pemikir kreatif.",
    llm=llm,
)

editor = Agent(
    role="Text Editor",
    goal="Memeriksa typo dan memperbaiki tata bahasa dari cerita.",
    backstory="Editor teliti.",
    llm=llm,
)

# Fungsi pembuat tugas (agar dapat direplikasi untuk 2 kru yang berbeda)
def create_tasks():
    t1 = Task(
        description="Buat satu paragraf cerita pendek tentang AI.",
        expected_output="Satu paragraf cerita AI.",
        agent=ideator,
    )
    t2 = Task(
        description="Edit cerita dari t1 agar lebih dramatis dan tanpa typo.",
        expected_output="Cerita yang telah diedit.",
        agent=editor,
        context=[t1]
    )
    return [t1, t2]

# 2. Set Up Crew Sequential
crew_seq = Crew(
    agents=[ideator, editor],
    tasks=create_tasks(),
    process=Process.sequential,
    verbose=False,
)

# 3. Set Up Crew Hierarchical
crew_hier = Crew(
    agents=[ideator, editor],  # Manager jangan dimasukkan ke daftar agents
    tasks=create_tasks(),
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=False,
)

if __name__ == "__main__":
    print("Mulai eksekusi Process.sequential...")
    start_seq = time.time()
    res_seq = crew_seq.kickoff()
    time_seq = time.time() - start_seq

    print("Mulai eksekusi Process.hierarchical...")
    start_hier = time.time()
    res_hier = crew_hier.kickoff()
    time_hier = time.time() - start_hier

    # 4. Dokumentasi & Perbandingan
    print("\n\n=== PERBANDINGAN HASIL ===")
    print(f"[1] Waktu Sequential   : {time_seq:.2f} detik")
    print(f"[2] Waktu Hierarchical : {time_hier:.2f} detik\n")
    
    print("--- Output Sequential ---")
    print(res_seq)
    print("\n--- Output Hierarchical ---")
    print(res_hier)

    print("\n=== KESIMPULAN PENGGUNAAN ===")
    print("- Sequential   : Lebih cepat, hemat biaya (token), dan cocok untuk alur kerja yang sudah pasti (linear).")
    print("- Hierarchical : Lebih lambat (karena manager harus merencanakan dan mendelegasi) tapi cocok untuk tugas kompleks yang butuh pengawasan kualitas dan delegasi dinamis.")
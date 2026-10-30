# Nama File: 14.7.1_ContentPipeline.py
import os
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi LLM
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# 1. Definisi Agen
researcher = Agent(
    role="Senior AI Researcher",
    goal="Mencari poin-poin utama mengenai Fine-Tuning vs RAG.",
    backstory="Peneliti AI berpengalaman yang ahli dalam mencari literatur teknis.",
    llm=llm,
    verbose=True,
)

outline_writer = Agent(
    role="Content Outline Writer",
    goal="Menyusun struktur postingan blog berdasarkan riset.",
    backstory="Spesialis pembuat kerangka tulisan (outline) yang logis dan terstruktur.",
    llm=llm,
    verbose=True,
)

full_writer = Agent(
    role="Technical Blog Writer",
    goal="Mengembangkan kerangka tulisan menjadi postingan blog yang lengkap.",
    backstory="Penulis konten teknis yang andal dalam menjelaskan konsep AI yang kompleks.",
    llm=llm,
    verbose=True,
)

# 2. Definisi Tasks
research_task = Task(
    description="Cari poin utama, perbedaan, kelebihan, dan kelemahan dari Fine-Tuning vs RAG.",
    expected_output="Daftar poin utama (bullet points) mengenai Fine-Tuning dan RAG.",
    agent=researcher,
)

outline_task = Task(
    description="Gunakan hasil riset untuk membuat struktur postingan blog 'The practical guide to fine-tuning vs RAG'.",
    expected_output="Kerangka blog (Outline) yang berisi judul dan sub-judul.",
    agent=outline_writer,
    context=[research_task],
)

write_task = Task(
    description="Tulis postingan blog lengkap berdasarkan outline yang diberikan. Gunakan bahasa yang mudah dipahami developer.",
    expected_output="Artikel blog teknis lengkap berformat Markdown.",
    agent=full_writer,
    context=[outline_task],
)

# 3. Eksekusi Crew
crew = Crew(
    agents=[researcher, outline_writer, full_writer],
    tasks=[research_task, outline_task, write_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n=== HASIL AKHIR POSTINGAN BLOG ===")
    print(result)
# Nama File: 14.7.2_CodeReviewCrew.py
import os
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# Snippet kode Python yang sengaja dibuat kurang efisien dan rawan (sebagai tes)
PYTHON_SNIPPET = """
def fetch_user(username):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Rentan SQL Injection
    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
    users = cursor.fetchall()
    
    # Loop yang tidak efisien
    result = []
    for u in users:
        for i in range(len(users)):
            if u == users[i]:
                result.append(u)
    return result
"""

# 1. Definisi Agen
developer = Agent(
    role="Software Developer",
    goal="Menulis dan menjelaskan kode Python awal.",
    backstory="Developer yang sedang belajar dan baru saja menulis fungsi akses database.",
    llm=llm,
)

security_reviewer = Agent(
    role="Security Reviewer",
    goal="Mencari celah keamanan (vulnerabilities) pada kode.",
    backstory="Pakar Cybersecurity yang ahli mendeteksi injeksi SQL dan celah keamanan lainnya.",
    llm=llm,
)

performance_reviewer = Agent(
    role="Performance Reviewer",
    goal="Mengecek efisiensi (inefficiency) algoritma dan struktur data.",
    backstory="Senior Engineer yang ahli mengoptimalkan Big O notation dan kecepatan runtime.",
    llm=llm,
)

lead = Agent(
    role="Lead Engineer",
    goal="Menggabungkan feedback keamanan dan performa menjadi laporan akhir.",
    backstory="Lead Engineer yang merangkum hasil code review untuk diberikan ke developer.",
    llm=llm,
)

# 2. Definisi Tasks
dev_task = Task(
    description=f"Jelaskan secara singkat fungsi dari snippet kode berikut:\n\n{PYTHON_SNIPPET}",
    expected_output="Penjelasan singkat mengenai apa yang dilakukan oleh kode tersebut.",
    agent=developer,
)

sec_task = Task(
    description="Periksa kode Python yang diberikan dari dev_task, temukan celah keamanannya.",
    expected_output="Daftar celah keamanan dan cara memperbaikinya.",
    agent=security_reviewer,
    context=[dev_task],
)

perf_task = Task(
    description="Periksa kode Python yang diberikan dari dev_task, temukan inefisiensinya (seperti nested loop).",
    expected_output="Daftar masalah performa dan cara optimasinya.",
    agent=performance_reviewer,
    context=[dev_task],
)

lead_task = Task(
    description="Gabungkan hasil temuan dari Security Reviewer dan Performance Reviewer menjadi satu laporan code review utuh.",
    expected_output="Laporan code review berformat Markdown dengan bagian Security dan Performance.",
    agent=lead,
    context=[sec_task, perf_task],
)

# 3. Eksekusi Crew
crew = Crew(
    agents=[developer, security_reviewer, performance_reviewer, lead],
    tasks=[dev_task, sec_task, perf_task, lead_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n=== LAPORAN FINAL CODE REVIEW ===")
    print(result)
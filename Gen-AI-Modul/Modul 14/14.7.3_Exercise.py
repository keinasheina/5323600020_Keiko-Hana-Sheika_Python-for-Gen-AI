# Nama File: 14.7.3_CustomTool.py
import os
from typing import Type
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Definisi Custom Tool
class WebSearchInput(BaseModel):
    query: str = Field(description="Kata kunci untuk pencarian web")

class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = "Cari informasi di web. Wajib gunakan ini jika butuh data eksternal."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        # Mock search function returning predefined results
        query_lower = query.lower()
        if "crewai" in query_lower:
            return "CrewAI adalah framework canggih untuk mengorkestrasi agen AI role-playing secara otonom."
        elif "openrouter" in query_lower:
            return "OpenRouter adalah antarmuka API terpadu yang menyediakan akses ke berbagai LLM."
        return f"Hasil pencarian mock untuk: {query}. Data tidak ditemukan di mock database."

search_tool = WebSearchTool()

# 2. Definisi Konfigurasi, Agen & Task
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

researcher = Agent(
    role="Research Agent",
    goal="Mencari jawaban akurat menggunakan WebSearchTool.",
    backstory="Asisten riset yang selalu menggunakan tools pencarian untuk memastikan fakta.",
    llm=llm,
    tools=[search_tool],  # Wire tool to the agent
    verbose=True,
)

search_task = Task(
    description="Cari tahu apa itu 'CrewAI' dan 'OpenRouter' menggunakan WebSearchTool, lalu rangkum hasilnya.",
    expected_output="Rangkuman singkat tentang CrewAI dan OpenRouter berdasarkan hasil pencarian.",
    agent=researcher,
)

# 3. Eksekusi Crew
crew = Crew(
    agents=[researcher],
    tasks=[search_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n=== HASIL JAWABAN RESEARCHER ===")
    print(result)
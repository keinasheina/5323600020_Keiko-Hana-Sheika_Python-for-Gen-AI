import os
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any
from dotenv import load_dotenv

import anthropic
from openai import OpenAI

load_dotenv()

# ==========================================
# Inisialisasi Client
# ==========================================

# 1. Anthropic Client (Qwen via Xkiro)
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="https://api.xkiro.com",
)

# 2. OpenAI Client (GPT-4o-mini via OpenRouter)
openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# ==========================================
# Latihan 1: Tiga Versi System Prompt (Code Review)
# ==========================================
# Prompt basic, intermediate, dan expert-level[cite: 16]

PROMPTS = {
    "basic": "You are a code reviewer. Look at this code and tell me if it is good or bad.",
    "intermediate": "You are a Python code reviewer. Check this code for bugs, readability, and suggest improvements.",
    "expert": """You are a Principal Security & Performance Engineer reviewing production code.
Identify vulnerabilities (e.g. OWASP), Big-O inefficiencies, and anti-patterns.
Format your response strictly as:
- Issue:
- Impact:
- Fix:"""
}

# 5 snippets kode bermasalah[cite: 16]
CODE_SNIPPETS = [
    "def append_to_list(val, my_list=[]):\n    my_list.append(val)\n    return my_list", # Mutable default
    "def get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id = {id}')",   # SQL Injection
    "def read_file(path):\n    f = open(path, 'r')\n    return f.read()",               # File not closed
    "try:\n    do_something()\nexcept Exception:\n    pass",                            # Silent exception
    "def find_dup(lst):\n    return [x for x in lst if lst.count(x) > 1]"               # O(N^2) complexity
]

def evaluate_code_reviewers():
    print("--- Latihan 1: Evaluasi System Prompts ---")
    # Menggunakan snippet ke-2 (SQL Injection) sebagai contoh demo agar output tidak terlalu panjang
    snippet = CODE_SNIPPETS[1] 
    print(f"Target Snippet:\n{snippet}\n")
    
    for level, sys_prompt in PROMPTS.items():
        resp = anthropic_client.messages.create(
            model="qwen/qwen3.8-max:free",
            max_tokens=256,
            system=sys_prompt,
            messages=[{"role": "user", "content": snippet}]
        )
        print(f"[{level.upper()}]\n{resp.content[0].text.strip()}\n")


# ==========================================
# Latihan 2: Class PromptLibrary
# ==========================================
# Menyimpan, load/save JSON, dan melacak versi evaluasi[cite: 16]

@dataclass
class PromptTemplate:
    name: str
    system: str
    version: str = "1.0"
    last_evaluated_at: str = "Never"

class PromptLibrary:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def add_template(self, template: PromptTemplate):
        self.templates[template.name] = template

    def mark_evaluated(self, name: str, timestamp: str):
        if name in self.templates:
            self.templates[name].last_evaluated_at = timestamp

    def save_to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            data = {name: asdict(tpl) for name, tpl in self.templates.items()}
            json.dump(data, f, indent=4)

    def load_from_json(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            for name, tpl_data in data.items():
                self.templates[name] = PromptTemplate(**tpl_data)

def test_prompt_library():
    print("--- Latihan 2: PromptLibrary ---")
    lib = PromptLibrary()
    lib.add_template(PromptTemplate("reviewer_basic", PROMPTS["basic"]))
    lib.mark_evaluated("reviewer_basic", "2026-09-29 10:00:00")
    
    lib.save_to_json("prompts.json")
    print("Berhasil menyimpan prompt ke 'prompts.json'.")
    
    new_lib = PromptLibrary()
    new_lib.load_from_json("prompts.json")
    print(f"Data dimuat: {new_lib.templates['reviewer_basic'].name}, Terakhir dievaluasi: {new_lib.templates['reviewer_basic'].last_evaluated_at}\n")


# ==========================================
# Latihan 3: Automatic JSON Repair
# ==========================================
# Fungsi safe_json_parse(text: str) -> dict[cite: 16]

def safe_json_parse(text: str) -> dict:
    # 1. Coba parse langsung
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Coba bersihkan markdown fences (```json ... ```)
    try:
        cleaned = re.search(r'```(?:json)?\n?(.*?)\n?```', text, re.DOTALL)
        if cleaned:
            return json.loads(cleaned.group(1).strip())
    except Exception:
        pass

    # 3. Minta bantuan LLM (OpenAI) untuk memperbaiki JSON yang rusak
    print(">> JSON rusak, meminta bantuan LLM untuk memperbaiki...")
    repair_prompt = f"Fix this broken JSON string and return ONLY valid JSON, no markdown formatting:\n{text}"
    
    resp = openai_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": repair_prompt}],
    )
    fixed_text = resp.choices[0].message.content.strip()
    
    # Strip markdown in case LLM ignored the instruction
    fixed_text = re.sub(r'^```(?:json)?\n?|\n?```$', '', fixed_text, flags=re.MULTILINE).strip()
    return json.loads(fixed_text)

def test_json_repair():
    print("--- Latihan 3: Automatic JSON Repair ---")
    broken_json = """```json
    {
        "name": "John"
        "age": 30,
        "is_active": true,
    }
    ```""" # Missing comma after John, trailing comma after true
    
    result = safe_json_parse(broken_json)
    print("Hasil perbaikan JSON:", result)
    print()


# ==========================================
# Latihan 4: Chain-of-Thought (CoT) Prompt
# ==========================================
# Memberikan ranking model LLM dari skor 3 task dan menulis 2 kalimat rekomendasi[cite: 16]

COT_SYSTEM = """Solve the task using this exact format:

<thinking>
1. Calculate the average score for each model across all 3 tasks.
2. Sort the models from highest to lowest average score.
3. Draft a 2-sentence recommendation based on the top model's strengths.
</thinking>

<answer>
1. [Model Name] - Avg: [Score]
2. [Model Name] - Avg: [Score]
...

Recommendation: [Your 2-sentence recommendation here]
</answer>"""

EVAL_INPUTS = [
    "ModelA (Math: 80, Coding: 90, Logic: 70) | ModelB (Math: 95, Coding: 85, Logic: 90)",
    "GPT-4 (MMLU: 88, HumanEval: 92, GSM8K: 95) | Llama-3 (MMLU: 82, HumanEval: 81, GSM8K: 85)",
    "ModelX (Task1: 50, Task2: 50, Task3: 50) | ModelY (Task1: 60, Task2: 40, Task3: 60) | ModelZ (Task1: 90, Task2: 90, Task3: 95)"
]

def test_cot_ranking():
    print("--- Latihan 4: CoT Prompt untuk LLM Ranking ---")
    for i, user_input in enumerate(EVAL_INPUTS, 1):
        print(f"\n[Test Case {i}] Input: {user_input}")
        
        resp = anthropic_client.messages.create(
            model="qwen/qwen3.8-max:free",
            max_tokens=512,
            system=COT_SYSTEM,
            messages=[{"role": "user", "content": user_input}]
        )
        
        text = resp.content[0].text
        # Ekstrak bagian answer saja agar rapi
        answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer:
            print(f"Hasil:\n{answer.group(1).strip()}")
        else:
            print("Gagal mengekstrak <answer>. Raw output:\n", text)


# ==========================================
# Eksekusi Utama
# ==========================================
if __name__ == "__main__":
    evaluate_code_reviewers()
    test_prompt_library()
    test_json_repair()
    test_cot_ranking()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset

# ── 1. Setup Base Model & Tokenizer ──────────────────────────────────────────
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Sangat penting: Paksa pindah ke CPU sejak awal untuk menghindari bug MPS Apple Silicon
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

# ── 2. Configure LoRA ────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
)
peft_model = get_peft_model(model, lora_config)

# ── 3. Custom Dataset (20 Q&A Pairs) ─────────────────────────────────────────
custom_qa_pairs = [
    "Question: Where are transaction logs displayed? Answer: They are displayed from the MySQL database.",
    "Question: Where is remaining stock calculated? Answer: Stock calculations are executed in the backend logic.",
    "Question: Does the system use wallet integration? Answer: No, there is no wallet integration or payment processing.",
    "Question: Which network is used for the simulation? Answer: Ganache local network is used.",
    "Question: What is the main function of the system? Answer: To handle stock-in and stock-out inputs.",
    "Question: Are logs fetched directly from the blockchain? Answer: No, they are fetched from the database.",
    "Question: What framework is used for the backend? Answer: The backend uses Node.js API.",
    "Question: Is the system deployed to a live public network? Answer: No, it is still a prototype and simulation.",
    "Question: How is the frontend architecture structured? Answer: It uses standalone HTML files with external CSS stylesheets.",
    "Question: What technology writes the transaction to the ledger? Answer: Solidity smart contracts.",
    "Question: Who can access the dashboard? Answer: Admin and Staff based on Role-Based Access Control.",
    "Question: Why is external CSS preferred? Answer: To separate structure in HTML from visual styling.",
    "Question: Does the warehouse staff need crypto to use it? Answer: No crypto or wallet is needed for inputs.",
    "Question: Where is the backend data stored? Answer: In a MySQL relational database.",
    "Question: Is this a final production system? Answer: No, it is not used directly by the real warehouse yet.",
    "Question: What handles the web interface? Answer: React.js and standalone HTML/CSS pages.",
    "Question: Are styles written inside HTML? Answer: No, styles are separated into external CSS files.",
    "Question: What happens when stock goes out? Answer: The backend calculates the remaining stock.",
    "Question: How is the UI/UX improved? Answer: By redesigning the layout for better user experience.",
    "Question: What environment simulates the blockchain? Answer: The Ganache local blockchain."
]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )

dataset = Dataset.from_dict({"text": custom_qa_pairs})
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})

# ── 4. Training Arguments ────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="/tmp/lora_custom",
    num_train_epochs=5,                 # Dinaikkan ke 5 agar hasilnya sedikit lebih baik
    per_device_train_batch_size=2,
    learning_rate=3e-4,
    fp16=False,
    logging_steps=5,
    save_strategy="no",                 # Tidak perlu disave setiap epoch untuk latihan ini
    report_to="none",
    use_cpu=True,                       # Kunci bypass error MPS Apple Silicon
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\nStarting LoRA Fine-Tuning on CPU...")
trainer.train()

peft_model.save_pretrained("/tmp/lora_custom_adapter")
print("\nAdapter saved. Loading for inference...")

# ── 5. Inference Verification ────────────────────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
merged = PeftModel.from_pretrained(base, "/tmp/lora_custom_adapter").to("cpu")
merged.eval()

# Mari kita tes dengan salah satu pertanyaan dari dataset
test_prompt = "Question: Does the system use wallet integration? Answer:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")

with torch.no_grad():
    out = merged.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)

print("\n--- INFERENCE RESULT ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))
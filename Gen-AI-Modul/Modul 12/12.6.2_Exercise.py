from transformers import AutoTokenizer

class TokenBudgetChecker:
    def __init__(self, model_name: str, token_limit: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.token_limit = token_limit

    def filter_texts(self, texts: list[str]) -> list[dict]:
        passed_texts = []
        for text in texts:
            # Hitung jumlah token dari teks
            token_count = len(self.tokenizer.encode(text))
            
            # Filter teks yang tidak melebihi batas (budget)
            if token_count <= self.token_limit:
                passed_texts.append({
                    "text": text,
                    "token_count": token_count
                })
        return passed_texts

if __name__ == "__main__":
    # Inisialisasi checker dengan batas maksimal 15 token
    checker = TokenBudgetChecker(model_name="distilgpt2", token_limit=15)
    
    sample_texts = [
        "Stock item added successfully.",
        "This sentence is intentionally made very long to test if the filter successfully removes it from the list.",
        "Query executed in backend.",
        "The blockchain transaction was verified and logged into the smart contract securely."
    ]
    
    print(f"Filtering texts with token limit <= {checker.token_limit}...\n")
    results = checker.filter_texts(sample_texts)
    
    for res in results:
        print(f"Count: {res['token_count']} | Text: {res['text']}")
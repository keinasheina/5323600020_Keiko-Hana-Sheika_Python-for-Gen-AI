import time
from transformers import pipeline

class ModelBenchmark:
    def __init__(self, model1_name="distilgpt2", model2_name="sshleifer/tiny-gpt2"):
        print("Loading models... (This might take a moment)")
        # Menggunakan dua model berukuran kecil agar cepat berjalan di CPU
        self.pipe1 = pipeline("text-generation", model=model1_name, device=-1)
        self.pipe2 = pipeline("text-generation", model=model2_name, device=-1)
        self.model1_name = model1_name
        self.model2_name = model2_name

    def run(self, prompts: list[str]):
        results = []
        print(f"\nRunning {len(prompts)} prompts...")
        
        for p in prompts:
            # Uji Model 1
            start_time = time.time()
            out1 = self.pipe1(p, max_new_tokens=20, pad_token_id=self.pipe1.tokenizer.eos_token_id)[0]['generated_text']
            lat1 = time.time() - start_time
            len1 = len(out1.split())

            # Uji Model 2
            start_time = time.time()
            out2 = self.pipe2(p, max_new_tokens=20, pad_token_id=self.pipe2.tokenizer.eos_token_id)[0]['generated_text']
            lat2 = time.time() - start_time
            len2 = len(out2.split())

            results.append({
                "prompt": p[:25] + "..." if len(p) > 25 else p,
                "m1_lat": lat1, "m1_len": len1,
                "m2_lat": lat2, "m2_len": len2
            })

        # Cetak Tabel Perbandingan
        print(f"\n| {'Prompt':<28} | {self.model1_name[:15]:<15} (Lat/Len) | {self.model2_name[:15]:<15} (Lat/Len) |")
        print("|" + "-"*30 + "|" + "-"*27 + "|" + "-"*27 + "|")
        for r in results:
            print(f"| {r['prompt']:<28} | {r['m1_lat']:.2f}s / {r['m1_len']:<4} tokens | {r['m2_lat']:.2f}s / {r['m2_len']:<4} tokens |")

if __name__ == "__main__":
    prompts = [
        "The advantage of modular",
        "Frontend architecture using",
        "Smart contracts are deployed",
        "When designing a UI/UX,",
        "To fetch data from an API",
        "A relational database like",
        "The backend server handles",
        "In a warehouse inventory",
        "Separating HTML and CSS",
        "To ensure transparency in"
    ]
    benchmark = ModelBenchmark()
    benchmark.run(prompts)
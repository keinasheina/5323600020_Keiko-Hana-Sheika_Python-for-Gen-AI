from transformers import pipeline

def extract_entities_from_corpus(texts: list[str]) -> dict:
    # Memuat pipeline NER. aggregation_strategy="simple" sangat penting agar 
    # entitas dengan multi-token digabungkan menjadi satu (contoh: San + Francisco)
    ner_pipeline = pipeline(
        task="ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=-1 
    )
    
    # Menggunakan Set untuk memastikan nilai yang disimpan unik (tidak duplikat)
    extracted = {"PER": set(), "ORG": set(), "LOC": set()}
    
    for text in texts:
        entities = ner_pipeline(text)
        for ent in entities:
            entity_group = ent['entity_group']
            # Masukkan ke set yang sesuai jika tipe entitas ada di dictionary kita
            if entity_group in extracted:
                extracted[entity_group].add(ent['word'])
                
    # Konversi set kembali ke list sesuai dengan requirement output
    return {key: list(values) for key, values in extracted.items()}

if __name__ == "__main__":
    corpus = [
        "Keiko is leading the UI/UX redesign project for PT Jawa Pos Media Televisi in Surabaya.",
        "Dario Amodei founded Anthropic, while Tim Cook runs Apple in Cupertino.",
        "She submitted her application to Samsung R&D Institute and PT Mayora Indah Tbk."
    ]
    
    print("Extracting entities...\n")
    result_dict = extract_entities_from_corpus(corpus)
    
    for entity_type, values in result_dict.items():
        print(f"{entity_type}: {values}")
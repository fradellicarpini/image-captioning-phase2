import json, os
from pathlib import Path

# === CONFIG ===
INPUT = Path("dataset_flickr8k_clean.json")   # il tuo file JSON pulito
OUTPUT_BASE = Path("data")                    # base_path del framework
DATASET_NAME = "Flickr8k"                     # nome cartella dataset
QUESTION = "Describe the image."              # prompt fisso
# ==============

out_dir = OUTPUT_BASE / DATASET_NAME
out_dir.mkdir(parents=True, exist_ok=True)

def tokens_to_sentence(tokens):
    s = " ".join(tokens).strip()
    # correzioni spazi/punteggiatura comuni
    s = (s.replace(" ,", ",")
           .replace(" .", ".")
           .replace(" !", "!")
           .replace(" ?", "?")
           .replace(" ;", ";")
           .replace(" :", ":")
           .replace(" '", "'")
           .replace(" n't", "n't"))
    # aggiungi un punto finale se manca
    if s and s[-1] not in ".!?":
        s += "."
    # maiuscola iniziale
    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

splits = {"train": [], "val": [], "test": []}
count_images = {"train": 0, "val": 0, "test": 0}
missing = 0

for item in data.get("images", []):
    fn = item.get("filename", "").strip()
    split = item.get("split", "").strip().lower()  # 'train' | 'val' | 'test'
    if split not in splits:
        continue
    count_images[split] += 1
    sentences = item.get("sentences", [])
    if not sentences:
        missing += 1
        continue
    for sent in sentences:
        tokens = sent.get("tokens", [])
        if not tokens:
            continue
        cap = tokens_to_sentence(tokens)
        splits[split].append({
            "image": fn,
            "question": QUESTION,
            "answer": cap,
            "need_external_knowledge": False
        })

# scrivi i file finali
for name in ["train", "val", "test"]:
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits[name], f, ensure_ascii=False, indent=2)

print("Conversione completata.")
print("Immagini per split:", count_images)
print("Esempi (caption espanse):", {k: len(v) for k, v in splits.items()})
print("Immagini senza caption:", missing)

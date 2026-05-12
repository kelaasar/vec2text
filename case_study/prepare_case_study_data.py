import json
import random
from pathlib import Path
from datasets import load_dataset

SAMPLE_SIZE = 800
MIN_LEN = 60
MAX_LEN = 80
random.seed(42)
OUT_DIR = Path("case_study_data")
OUT_DIR.mkdir(exist_ok=True)

print("Loading One Million Instructions...")
one_mil = load_dataset("wentingzhao/one-million-instructions", split="train")
all_instructions = [item["user"] for item in one_mil
                    if isinstance(item["user"], str) and MIN_LEN <= len(item["user"]) <= MAX_LEN]
random.shuffle(all_instructions)

known = all_instructions[:SAMPLE_SIZE]
Path(OUT_DIR / "known_text.json").write_text(json.dumps(known, indent=2))
print(f"  known_text: {len(known)} samples saved")

medical_keywords = ["medical", "health", "disease", "treatment", "patient",
                    "doctor", "medicine", "symptom", "diagnosis", "hospital",
                    "drug", "clinical", "surgery", "therapy", "vaccine"]
medical = [item["user"] for item in one_mil
           if isinstance(item["user"], str) and MIN_LEN <= len(item["user"]) <= MAX_LEN and
           any(kw in item["user"].lower() for kw in medical_keywords)]
random.shuffle(medical)
medical = medical[:SAMPLE_SIZE]
Path(OUT_DIR / "medical_text.json").write_text(json.dumps(medical, indent=2))
print(f"  medical_text: {len(medical)} samples saved")

print("Loading Wikitext-103...")
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
wiki_texts = [item["text"].strip() for item in wiki
              if isinstance(item["text"], str) and MIN_LEN <= len(item["text"].strip()) <= MAX_LEN
              and not item["text"].startswith(" =")]
random.shuffle(wiki_texts)
unknown = wiki_texts[:SAMPLE_SIZE]
Path(OUT_DIR / "unknown_text.json").write_text(json.dumps(unknown, indent=2))
print(f"  unknown_text: {len(unknown)} samples saved")

print("Loading PAWS...")
paws = load_dataset("paws", "labeled_final", split="train")
paraphrases = [item["sentence1"] for item in paws
               if item["label"] == 1 and MIN_LEN <= len(item["sentence1"]) <= MAX_LEN]
random.shuffle(paraphrases)
paraphrases = paraphrases[:SAMPLE_SIZE]
Path(OUT_DIR / "paraphrase_text.json").write_text(json.dumps(paraphrases, indent=2))
print(f"  paraphrase_text: {len(paraphrases)} samples saved")

code_keywords = ["code", "python", "javascript", "function", "algorithm",
                 "program", "script", "sql", "database", "implement", "debug",
                 "class", "variable", "loop", "array", "string", "integer"]
code_texts = [item["user"] for item in one_mil
              if isinstance(item["user"], str) and MIN_LEN <= len(item["user"]) <= MAX_LEN and
              any(kw in item["user"].lower() for kw in code_keywords)]
random.shuffle(code_texts)
code = code_texts[:SAMPLE_SIZE]
Path(OUT_DIR / "code.json").write_text(json.dumps(code, indent=2))
print(f"  code: {len(code)} samples saved")

print("Done. All files saved to case_study_data/")

import json
import os
from pathlib import Path

import torch
import evaluate
import vec2text
from vec2text.models import InversionModel, CorrectorEncoderModel
from vec2text import load_corrector
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("case_study_data")

if not DATA_DIR.exists():
    raise RuntimeError("case_study_data/ not found. Run prepare_case_study_data.py first.")

print("Loading case study data from local files...")
KNOWN_TEXTS     = json.loads((DATA_DIR / "known_text.json").read_text())
MEDICAL_TEXTS   = json.loads((DATA_DIR / "medical_text.json").read_text())
UNKNOWN_TEXTS   = json.loads((DATA_DIR / "unknown_text.json").read_text())
PARAPHRASE_TEXTS = json.loads((DATA_DIR / "paraphrase_text.json").read_text())
CODE_SNIPPETS   = json.loads((DATA_DIR / "code.json").read_text())

print(f"  known_text:      {len(KNOWN_TEXTS)} samples")
print(f"  medical_text:    {len(MEDICAL_TEXTS)} samples")
print(f"  unknown_text:    {len(UNKNOWN_TEXTS)} samples")
print(f"  paraphrase_text: {len(PARAPHRASE_TEXTS)} samples")
print(f"  code:            {len(CODE_SNIPPETS)} samples")

MODEL_CONFIGS = {
    "GTR-base": {
        "inversion": "saves/gtr-1/checkpoint-34194",
        "corrector": "saves/gtr-corrector-4gpu-2epochs/checkpoint-68386",
    },
    "text-embedding-3-small": {
        "inversion": "saves/openai-3small-inverter-fixed/checkpoint-136772",
        "corrector": "saves/openai-3small-corrector-fixed/checkpoint-72912",
    },
    "mistral-embed": {
        "inversion": "saves/mistral-embed-inverter/checkpoint-45592",
        "corrector": "saves/mistral-embed-corrector/checkpoint-91010",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_corrector_for(model_name):
    cfg = MODEL_CONFIGS[model_name]
    inv = InversionModel.from_pretrained(cfg["inversion"]).to(DEVICE)
    cor = CorrectorEncoderModel.from_pretrained(cfg["corrector"]).to(DEVICE)
    corrector = load_corrector(inv, cor)
    return corrector


def model_requires_api(model_name):
    if model_name == "text-embedding-3-small":
        return "OPENAI_API_KEY"
    if model_name == "mistral-embed":
        return "MISTRAL_API_KEY"
    return None


def exact_match(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())


def main():
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    case_studies = {
        "known_text": KNOWN_TEXTS,
        "unknown_text": UNKNOWN_TEXTS,
        "paraphrase_text": PARAPHRASE_TEXTS,
        "medical_text": MEDICAL_TEXTS,
        "code": CODE_SNIPPETS,
    }

    results = {}
    summary = {}

    for model_name in MODEL_CONFIGS:
        api_key_name = model_requires_api(model_name)
        if api_key_name is not None and os.getenv(api_key_name) is None:
            print(f"SKIPPING {model_name} because {api_key_name} is not set.")
            continue

        print(f"Loading model {model_name}...")
        corrector = load_corrector_for(model_name)
        print(f"Loaded {model_name}")

        model_results = {}
        for case, texts in case_studies.items():
            print(f"Running case {case} for {model_name} ({len(texts)} samples)")
            preds = []
            for text in texts:
                out = vec2text.invert_strings([text], corrector, num_steps=5)
                preds.append(out[0])
            ems = [exact_match(p, r) for p, r in zip(preds, texts)]
            bleu_res = bleu.compute(predictions=preds, references=[[r] for r in texts])
            rouge_res = rouge.compute(predictions=preds, references=texts)
            bert_res = bertscore.compute(predictions=preds, references=texts, lang="en")

            model_results[case] = {
                "exact_match": sum(ems) / len(ems),
                "bleu": bleu_res["bleu"],
                "rouge_l": rouge_res["rougeL"],
                "bertscore_f1": sum(bert_res["f1"]) / len(bert_res["f1"]),
            }
            m = model_results[case]
            print(f"  {case} EM={m['exact_match']:.3f} BLEU={m['bleu']:.3f} ROUGE-L={m['rouge_l']:.3f} BERTScore={m['bertscore_f1']:.3f}")

        results[model_name] = model_results

    for model_name, model_results in results.items():
        summary[model_name] = {case: metrics for case, metrics in model_results.items()}

    out = {"results": results, "summary": summary}
    Path("inversion_case_study_results_800.json").write_text(json.dumps(out, indent=2))
    print("Wrote inversion_case_study_results_800.json")


if __name__ == "__main__":
    main()

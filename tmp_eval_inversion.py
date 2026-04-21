import json
import os
from pathlib import Path
import random

import torch
import evaluate
import vec2text
from vec2text.models import InversionModel, CorrectorEncoderModel
from vec2text import load_corrector
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

# Load the One Million Instructions dataset
print("Loading One Million Instructions dataset...")
dataset = load_dataset("wentingzhao/one-million-instructions", split="train")
print(f"Loaded {len(dataset)} samples from dataset")

# Sample 800 items for each case category
SAMPLE_SIZE = 800
random.seed(42)

# Sample general instructions for known/unknown/paraphrase text
all_instructions = [item['instruction'] for item in dataset if isinstance(item['instruction'], str) and len(item['instruction']) > 10]
random.shuffle(all_instructions)

KNOWN_TEXTS = all_instructions[:SAMPLE_SIZE]
UNKNOWN_TEXTS = all_instructions[SAMPLE_SIZE:2*SAMPLE_SIZE]
PARAPHRASE_TEXTS = all_instructions[2*SAMPLE_SIZE:3*SAMPLE_SIZE]

# For medical text, try to filter for medical/healthcare related items
medical_keywords = ['medical', 'health', 'disease', 'treatment', 'patient', 'doctor', 'medicine', 'symptom', 'diagnosis', 'hospital']
medical_texts = [item['instruction'] for item in dataset 
                  if isinstance(item['instruction'], str) and 
                  any(keyword in item['instruction'].lower() for keyword in medical_keywords)]

if len(medical_texts) < SAMPLE_SIZE:
    # If not enough medical texts, pad with general instructions
    medical_texts.extend(all_instructions[3*SAMPLE_SIZE:3*SAMPLE_SIZE + (SAMPLE_SIZE - len(medical_texts))])

MEDICAL_TEXTS = medical_texts[:SAMPLE_SIZE]

# For code, look for code-related instructions
code_keywords = ['code', 'python', 'javascript', 'function', 'algorithm', 'program', 'script', 'sql', 'database']
code_texts = [item['instruction'] for item in dataset 
              if isinstance(item['instruction'], str) and 
              any(keyword in item['instruction'].lower() for keyword in code_keywords)]

if len(code_texts) < SAMPLE_SIZE:
    code_texts.extend(all_instructions[3*SAMPLE_SIZE:3*SAMPLE_SIZE + (SAMPLE_SIZE - len(code_texts))])

CODE_SNIPPETS = code_texts[:SAMPLE_SIZE]

print(f"Sampled {SAMPLE_SIZE} items for each case category")
print(f"  known_text: {len(KNOWN_TEXTS)} samples")
print(f"  unknown_text: {len(UNKNOWN_TEXTS)} samples")
print(f"  paraphrase_text: {len(PARAPHRASE_TEXTS)} samples")
print(f"  medical_text: {len(MEDICAL_TEXTS)} samples")
print(f"  code_snippets: {len(CODE_SNIPPETS)} samples")

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    "The only thing we have to fear is fear itself.",
    "In the beginning God created the heavens and the earth.",
    "All animals are equal, but some animals are more equal than others.",
    "A journey of a thousand miles begins with a single step.",
    "The quick brown fox jumps over the lazy dog.",
    "May the Force be with you.",
    "Elementary, my dear Watson.",
    "Houston, we have a problem.",
    "I have a dream that one day this nation will rise up.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The lady doth protest too much, methinks.",
    "You must be the change you wish to see in the world.",
    "That's one small step for man, one giant leap for mankind.",
    "The only constant in life is change.",
    "Not everything that is faced can be changed, but nothing can be changed until it is faced.",
    "All the world’s a stage, and all the men and women merely players.",
    "It does not do to dwell on dreams and forget to live.",
    "The truth is rarely pure and never simple.",
    "The course of true love never did run smooth.",
    "Life is what happens when you're busy making other plans.",
    "If you want to live a happy life, tie it to a goal, not to people or things.",
    "In three words I can sum up everything I've learned about life: it goes on.",
    "Not all those who wander are lost.",
    "The future belongs to those who believe in the beauty of their dreams.",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "You only live once, but if you do it right, once is enough.",
    "If you tell the truth, you don't have to remember anything.",
]

UNKNOWN_TEXTS = [
    "The matte purple kettle hummed quietly beneath a constellation of paper cranes.",
    "A tiny paper moth circled the bookshelf as the wind carried the scent of distant rain.",
    "Her crystal sneakers left faint footprints in the chalk-dusted courtyard.",
    "The turquoise river sang to the lanterns while the design team sketched new constellations.",
    "A lunchbox full of hummingbird postcards arrived at the moonlit studio.",
    "The staircase unfolded into a secret garden made of folded paper and starlight.",
    "He wrote recipes for invisible breakfasts that tasted like nostalgia.",
    "A slow thunder of clay pebbles echoed through the midnight museum.",
    "She found a library where the books whispered in color instead of words.",
    "The autumn train carried a choir of glass marbles across the twilight valley.",
    "A holographic dandelion floated over the room and released sentences instead of seeds.",
    "The clock on the roof ticked in oval rhythms while the rain painted the alley gold.",
    "He sketched dreams on the underside of a paper airplane and watched them fly away.",
    "A small robot made breakfast for the shadows at the edge of the city.",
    "The circus tent was woven from memories of forgotten summers.",
    "The musician tuned his violin to the language of folded letters.",
    "A bowl of violet soup appeared on the table, steaming with stories of distant islands.",
    "She wore a necklace made from tiny clocks that only chimed at midday.",
    "The photographer collected laughter in glass jars and labeled them by season.",
    "A flock of paper cranes built a bridge across the attic rafters.",
    "The canyon hummed with recipes for starlight pancakes.",
    "The garden was full of lanterns shaped like planets and leaves made of poetry.",
    "He planted photographs in the soil and grew a forest of silent afternoons.",
    "The antique shop sold time by the minute, wrapped in ribbon and cinnamon.",
    "A comet delivered a letter written in handwriting made of rain.",
    "The desk drawer opened to reveal a whispered conversation from the future.",
    "She danced with her shadow until the moon turned to paper.",
    "The lighthouse sent out signals in the form of folded origami boats.",
    "A cloud of lanterns floated above the city, each one carrying a restless memory.",
]

PARAPHRASE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaped over a sleepy dog.",
    "She began to read once the sun rose.",
    "When morning came, she opened her book.",
    "The crowd cheered loudly at the match.",
    "Fans erupted with applause during the game.",
    "He solved the problem in a single try.",
    "On his first attempt, he found the answer.",
    "The meeting was postponed until next Thursday.",
    "They delayed the session to the following Thursday.",
    "The storm knocked out power across the city.",
    "Electricity went down all over town because of the storm.",
    "She learned French during her year abroad.",
    "While studying overseas, she picked up French.",
    "The package arrived earlier than expected.",
    "It showed up before they thought it would.",
    "He carefully painted the old wooden fence.",
    "With care, he applied paint to the weathered fence.",
    "The restaurant closed after midnight.",
    "It shut its doors later than midnight.",
    "He studied the recipe before starting.",
    "Before he began, he read through the recipe.",
    "The artist mixed bright colors on the canvas.",
    "On the canvas, the painter blended vivid hues.",
    "She wrote the letter with a steady hand.",
    "Her handwriting stayed steady as she composed the note.",
    "The child built a sandcastle on the beach.",
    "At the shore, the kid made a castle from sand.",
    "The team celebrated their victory.",
    "They rejoiced after winning the match.",
]

MEDICAL_TEXTS = [
    "The patient presented with acute chest pain and shortness of breath.",
    "Her blood pressure was elevated, and she reported dizziness.",
    "The MRI scan showed a small lesion in the temporal lobe.",
    "He was prescribed a course of antibiotics for the suspected infection.",
    "The physician noted a significant reduction in inflammation.",
    "She has a history of type 2 diabetes and hypertension.",
    "The surgeon recommended minimally invasive laparoscopic surgery.",
    "A follow-up appointment was scheduled in two weeks.",
    "The medication caused mild nausea and headache.",
    "The laboratory results showed elevated white blood cell counts.",
    "The patient was advised to avoid strenuous exercise during recovery.",
    "He experienced intermittent abdominal pain after meals.",
    "The ECG indicated an irregular heartbeat rhythm.",
    "She received a vaccine booster to maintain immunity.",
    "The X-ray confirmed a fracture of the distal radius.",
    "The doctor reviewed the patient’s medical history in detail.",
    "The treatment plan included physical therapy and diet modification.",
    "He was referred to a specialist for further evaluation.",
    "The wound site was cleaned and dressed with sterile gauze.",
    "The nurse monitored the patient’s vital signs hourly.",
    "The condition improved after administration of intravenous fluids.",
    "The patient reported persistent fatigue and weakness.",
    "The clinical team discussed the risks and benefits of the procedure.",
    "The chest X-ray revealed no evidence of pneumonia.",
    "An allergy to penicillin was documented in the chart.",
    "The patient was placed on a low-sodium diet.",
    "She underwent a blood glucose test before breakfast.",
    "The specialist ordered additional imaging studies.",
    "The patient experienced relief after the pain medication.",
]

CODE_SNIPPETS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "for i in range(10):\n    print(i * i)",
    "class Dog:\n    def __init__(self, name):\n        self.name = name\n\n    def bark(self):\n        return f'{self.name} says woof!'\n",
    "import os\nfiles = [f for f in os.listdir('.') if f.endswith('.py')]\nprint(files)",
    "const add = (a, b) => a + b;\nconsole.log(add(3, 4));",
    "SELECT name, age FROM users WHERE active = TRUE ORDER BY age DESC;",
    "function greet(name) {\n  return `Hello, ${name}!`;\n}",
    "let arr = [1, 2, 3, 4];\nlet doubled = arr.map(x => x * 2);",
    "with open('log.txt', 'a') as f:\n    f.write('started\n')",
    "try:\n    value = int(input('Enter a number: '))\nexcept ValueError:\n    print('Invalid input')",
    "def merge_dicts(a, b):\n    return {**a, **b}",
    "const fetchData = async (url) => {\n  const res = await fetch(url);\n  return res.json();\n};",
    "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price DECIMAL(10,2));",
    "for (let i = 0; i < items.length; i++) {\n  console.log(items[i]);\n}",
    "def normalize(text):\n    return text.strip().lower()",
    "function factorial(n) {\n  if (n <= 1) return 1;\n  return n * factorial(n-1);\n}",
    "INSERT INTO orders (user_id, total) VALUES (123, 45.67);",
    "try {\n  const result = await compute();\n  console.log(result);\n} catch (e) {\n  console.error(e);\n}",
    "def flatten(list_of_lists):\n    return [item for sublist in list_of_lists for item in sublist]",
    "const user = { name: 'Alice', age: 30 };\nconsole.log(user.name);",
    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = 42;",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "const numbers = [1, 2, 3];\nconst sum = numbers.reduce((a, b) => a + b, 0);",
    "SELECT COUNT(*) FROM visits WHERE date >= '2026-01-01';",
    "def greet_all(names):\n    for name in names:\n        print(f'Hello, {name}')",
    "function filterEven(arr) {\n  return arr.filter(x => x % 2 === 0);\n}",
    "delete from sessions where expires < NOW();",
    "def compress(data):\n    return ''.join(sorted(set(data), key=data.index))",
    "let sum = 0;\nfor (let num of numbers) {\n  sum += num;\n}",
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_corrector_for(model_name):
    cfg = MODEL_CONFIGS[model_name]
    inv = InversionModel.from_pretrained(cfg['inversion']).to(DEVICE)
    cor = CorrectorEncoderModel.from_pretrained(cfg['corrector']).to(DEVICE)
    corrector = load_corrector(inv, cor)
    return corrector


def model_requires_api(model_name):
    if model_name == 'text-embedding-3-small':
        return 'OPENAI_API_KEY'
    if model_name == 'mistral-embed':
        return 'MISTRAL_API_KEY'
    return None


def exact_match(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())


def main():
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    case_studies = {
        'known_text': KNOWN_TEXTS,
        'unknown_text': UNKNOWN_TEXTS,
        'paraphrase_text': PARAPHRASE_TEXTS,
        'medical_text': MEDICAL_TEXTS,
        'code': CODE_SNIPPETS,
    }

    results = {}
    summary = {}

    for model_name in MODEL_CONFIGS:
        api_key_name = model_requires_api(model_name)
        if api_key_name is not None and os.getenv(api_key_name) is None:
            print(f'SKIPPING {model_name} because {api_key_name} is not set.')
            continue

        print(f'Loading model {model_name}...')
        corrector = load_corrector_for(model_name)
        print(f'Loaded {model_name}')

        model_results = {}
        for case, texts in case_studies.items():
            print(f'Running case {case} for {model_name} ({len(texts)} samples)')
            preds = []
            for text in texts:
                out = vec2text.invert_strings([text], corrector, num_steps=5)
                preds.append(out[0])
            ems = [exact_match(p, r) for p, r in zip(preds, texts)]
            bleu_res = bleu.compute(predictions=preds, references=[[r] for r in texts])
            rouge_res = rouge.compute(predictions=preds, references=texts)
            bert_res = bertscore.compute(predictions=preds, references=texts, lang='en')

            model_results[case] = {
                'exact_match': sum(ems) / len(ems),
                'bleu': bleu_res['bleu'],
                'rouge_l': rouge_res['rougeL'],
                'bertscore_f1': sum(bert_res['f1']) / len(bert_res['f1']),
            }
            print(f'  {case} EM={model_results[case]["exact_match"]:.3f} BLEU={model_results[case]["bleu"]:.3f} ROUGE-L={model_results[case]["rouge_l"]:.3f} BERTScore={model_results[case]["bertscore_f1"]:.3f}')

        results[model_name] = model_results

    for model_name, model_results in results.items():
        summary[model_name] = {
            case: {
                'exact_match': metrics['exact_match'],
                'bleu': metrics['bleu'],
                'rouge_l': metrics['rouge_l'],
                'bertscore_f1': metrics['bertscore_f1'],
            }
            for case, metrics in model_results.items()
        }

    out = {'results': results, 'summary': summary}
    Path('inversion_case_study_results.json').write_text(json.dumps(out, indent=2))
    print('Wrote inversion_case_study_results.json')


if __name__ == '__main__':
    main()

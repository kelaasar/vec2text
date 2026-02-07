"""
Interactive script to test your trained corrector model.
Input any sentence and see the results at each correction step.
"""

import os
import torch
import vec2text
from vec2text.models import InversionModel, CorrectorEncoderModel

# Set cache directory
os.environ['VEC2TEXT_CACHE'] = '/scratch/kelaasar/vec2text_cache'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODELS = {
    "1": {
        "name": "GTR-base",
        "inversion": "/scratch/kelaasar/vec2text/saves/gtr-1/checkpoint-34194",
        "corrector": "/scratch/kelaasar/vec2text/saves/gtr-corrector-4gpu-2epochs"
    },
    "2": {
        "name": "text-embedding-3-small (OpenAI)",
        "inversion": "/scratch/kelaasar/vec2text/saves/openai-3small-inverter-fixed/checkpoint-136772",
        "corrector": "/scratch/kelaasar/vec2text/saves/openai-3small-corrector-fixed"
    },
    "3": {
        "name": "gemini-embedding-001 (Google)",
        "inversion": "/scratch/kelaasar/vec2text/saves/gemini-embedding-001-inverter",
        "corrector": "/scratch/kelaasar/vec2text/saves/gemini-embedding-001-corrector"
    },
    "4": {
        "name": "mistral-embed (Mistral AI)",
        "inversion": "/scratch/kelaasar/vec2text/saves/mistral-embed-inverter/checkpoint-45592",
        "corrector": "/scratch/kelaasar/vec2text/saves/mistral-embed-corrector/checkpoint-91010"
    }
}

# Predefined test examples
PREDEFINED_EXAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is revolutionizing artificial intelligence.",
    "Python is a versatile programming language used in data science.",
    "Climate change is affecting global weather patterns.",
    "Quantum computing could transform cryptography and optimization.",
    "The stock market experienced significant volatility today.",
    "Renewable energy sources are becoming more cost-effective.",
    "Neural networks can recognize patterns in complex data.",
    "Space exploration continues to push the boundaries of human knowledge.",
    "Effective communication is essential for team collaboration."
]

# Let user choose model
print("="*70)
print("📊 AVAILABLE MODELS")
print("="*70)
for key, model_info in MODELS.items():
    print(f"  {key}. {model_info['name']}")
print("="*70)

while True:
    choice = input("\n➤ Select model (1, 2, 3, or 4): ").strip()
    if choice in MODELS:
        break
    print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

selected_model = MODELS[choice]
print(f"\n✅ Selected: {selected_model['name']}")

# Ask for test mode
print("\n" + "="*70)
print("🧪 TEST MODE")
print("="*70)
print("  1. Run 10 predefined examples (batch mode)")
print("  2. Interactive mode (enter your own text)")
print("  3. Load embedding from file or paste directly")
print("="*70)

while True:
    mode_choice = input("\n➤ Select test mode (1, 2, or 3): ").strip()
    if mode_choice in ["1", "2", "3"]:
        break
    print("❌ Invalid choice. Please enter 1, 2, or 3.")

use_predefined = (mode_choice == "1")
use_embedding_input = (mode_choice == "3")

# Handle embedding input mode
if use_embedding_input:
    print("\n" + "="*70)
    print("📥 EMBEDDING INPUT")
    print("="*70)
    print("  1. Load from text file")
    print("  2. Paste embedding directly")
    print("="*70)
    
    while True:
        embedding_input_mode = input("\n➤ Select input method (1 or 2): ").strip()
        if embedding_input_mode in ["1", "2"]:
            break
        print("❌ Invalid choice. Please enter 1 or 2.")
    
    embedding_values = None
    if embedding_input_mode == "1":
        filepath = input("\n➤ Enter path to embedding text file: ").strip()
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                embedding_values = [float(x) for x in content.split(',')]
            print(f"✅ Loaded {len(embedding_values)} values from {filepath}")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            exit(1)
    else:  # mode 2 - paste directly
        print("\n➤ Paste embedding (comma-separated values):")
        content = input().strip()
        try:
            embedding_values = [float(x) for x in content.split(',')]
            print(f"✅ Loaded {len(embedding_values)} values")
        except Exception as e:
            print(f"❌ Error parsing embedding: {e}")
            exit(1)
    
    # Verify dimensions
    if 'gtr' in selected_model['name'].lower() or 'gemini' in selected_model['name'].lower():
        expected_dim = 768
    elif 'mistral' in selected_model['name'].lower():
        expected_dim = 1024
    else:
        expected_dim = 1536
    if len(embedding_values) != expected_dim:
        print(f"\n⚠️  Warning: Expected {expected_dim} dimensions for {selected_model['name']} but got {len(embedding_values)}")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            exit(1)

# Ask for maximum number of correction steps
print("\n" + "="*70)
print("🔢 CORRECTION STEPS")
print("="*70)
print("Enter the maximum number of correction steps to test.")
print("The program will show results from 0 steps up to your chosen number.")
print("(Recommended: 2-10, but you can choose any positive number)")
print("="*70)

while True:
    try:
        max_steps = input("\n➤ Enter max correction steps (e.g., 5): ").strip()
        max_steps = int(max_steps)
        if max_steps >= 0:
            break
        print("❌ Please enter a non-negative number.")
    except ValueError:
        print("❌ Invalid input. Please enter a number.")

print(f"\n✅ Will test from 0 to {max_steps} correction steps")

# Load models (only once)
print("="*70)
print("Loading your trained corrector model...")
print("="*70)

inversion_checkpoint = selected_model["inversion"]
corrector_checkpoint = selected_model["corrector"]

print(f"Loading inversion model from: {inversion_checkpoint}")
inversion_model = InversionModel.from_pretrained(inversion_checkpoint)

print(f"Loading corrector model from: {corrector_checkpoint}")
corrector_model = CorrectorEncoderModel.from_pretrained(corrector_checkpoint)

print("\nCreating corrector trainer...")
corrector = vec2text.load_corrector(inversion_model, corrector_model)
corrector.model.to(device)
corrector.inversion_trainer.model.to(device)

print("\n✅ Models loaded successfully!")
print(f"🎯 Using: {selected_model['name']}")
print("="*70)

def invert_text(text, max_steps):
    """Invert text and return results at each step from 0 to max_steps."""
    results = {}
    
    for step in range(max_steps + 1):
        inverted = vec2text.invert_strings(
            strings=[text],
            corrector=corrector,
            num_steps=step,
        )[0]
        results[step] = inverted
    
    return results

def print_results(text, results):
    """Pretty print the results."""
    print("\n" + "="*70)
    print(f"ORIGINAL: '{text}'")
    print("="*70)
    
    for step in sorted(results.keys()):
        step_label = f"[{step} step{'s' if step != 1 else ''}]" if step > 0 else "[0 steps]"
        print(f"\n{step_label} {results[step]}")
    print("="*70)

# Run tests based on selected mode
if use_embedding_input:
    # Embedding input mode
    print("\n🔢 EMBEDDING INPUT MODE")
    print("="*70)
    print("Running inversion from provided embedding...")
    print("="*70)
    
    # Convert embedding values to tensor
    import torch
    embedding_tensor = torch.tensor([embedding_values], dtype=torch.float32).to(device)
    
    # Run the corrector at different steps
    print(f"\n⏳ Processing with 0 to {max_steps} correction steps...")
    print("\n" + "="*70)
    print("RESULTS FROM PROVIDED EMBEDDING")
    print("="*70)
    
    for step in range(max_steps + 1):
        outputs = vec2text.invert_embeddings(
            embeddings=embedding_tensor,
            corrector=corrector,
            num_steps=step
        )
        step_label = f"[{step} step{'s' if step != 1 else ''}]" if step > 0 else "[0 steps]"
        print(f"\n{step_label} {outputs[0]}")
    
    print("\n" + "="*70)
    print("✅ Embedding inversion completed!")
    print("="*70)

elif use_predefined:
    # Batch mode with predefined examples
    print("\n🧪 BATCH TEST MODE")
    print("="*70)
    print(f"Testing {len(PREDEFINED_EXAMPLES)} predefined examples...")
    print("="*70)
    
    for i, text in enumerate(PREDEFINED_EXAMPLES, 1):
        print(f"\n[Example {i}/{len(PREDEFINED_EXAMPLES)}]")
        print("⏳ Processing...")
        results = invert_text(text, max_steps)
        print_results(text, results)
    
    print("\n" + "="*70)
    print("✅ Batch testing completed!")
    print("="*70)
    
else:
    # Interactive loop
    print("\n🎤 INTERACTIVE MODE")
    print("="*70)
    print("Enter sentences to test (type 'quit' or 'exit' to stop)")
    print("="*70)

    while True:
        try:
            user_input = input("\n➤ Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                print("⚠️  Please enter some text.")
                continue
            
            print("\n⏳ Processing...")
            results = invert_text(user_input, max_steps)
            print_results(user_input, results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

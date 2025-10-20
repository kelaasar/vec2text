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

# Load models (only once)
print("="*70)
print("Loading your trained corrector model...")
print("="*70)

inversion_checkpoint = "/scratch/kelaasar/vec2text/saves/gtr-1/checkpoint-34194"
corrector_checkpoint = "/scratch/kelaasar/vec2text/saves/gtr-corrector-4gpu-2epochs"

print(f"Loading inversion model from: {inversion_checkpoint}")
inversion_model = InversionModel.from_pretrained(inversion_checkpoint)

print(f"Loading corrector model from: {corrector_checkpoint}")
corrector_model = CorrectorEncoderModel.from_pretrained(corrector_checkpoint)

print("\nCreating corrector trainer...")
corrector = vec2text.load_corrector(inversion_model, corrector_model)
corrector.model.to(device)
corrector.inversion_trainer.model.to(device)

print("\n✅ Models loaded successfully!")
print("="*70)

def invert_text(text):
    """Invert text and return results at each step."""
    # Get hypothesis (0 steps)
    hypothesis_text = vec2text.invert_strings(
        strings=[text],
        corrector=corrector,
        num_steps=0,
    )[0]
    
    # After 1 correction step
    corrected_1_text = vec2text.invert_strings(
        strings=[text],
        corrector=corrector,
        num_steps=1,
    )[0]
    
    # After 2 correction steps
    corrected_2_text = vec2text.invert_strings(
        strings=[text],
        corrector=corrector,
        num_steps=2,
    )[0]
    
    # After 5 correction steps
    corrected_5_text = vec2text.invert_strings(
        strings=[text],
        corrector=corrector,
        num_steps=5,
    )[0]
    
    return {
        'hypothesis': hypothesis_text,
        'step_1': corrected_1_text,
        'step_2': corrected_2_text,
        'step_5': corrected_5_text,
    }

def print_results(text, results):
    """Pretty print the results."""
    print("\n" + "="*70)
    print(f"ORIGINAL: '{text}'")
    print("="*70)
    
    print(f"\n[0 steps] {results['hypothesis']}")
    print(f"[1 step]  {results['step_1']}")
    print(f"[2 steps] {results['step_2']}")
    print(f"[5 steps] {results['step_5']}")
    print("="*70)

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
        results = invert_text(user_input)
        print_results(user_input, results)
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

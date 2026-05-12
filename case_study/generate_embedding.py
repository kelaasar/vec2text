"""
Script to generate embeddings from text using different embedding models.
Outputs the embedding as a numpy array that can be saved or used directly.
"""

import os
import torch
import numpy as np
from vec2text.models import InversionModel

# Set cache directory
os.environ['VEC2TEXT_CACHE'] = '/scratch/kelaasar/vec2text_cache'

# Available embedding models
EMBEDDING_MODELS = {
    "1": {
        "name": "GTR-base",
        "model_path": "/scratch/kelaasar/vec2text/saves/gtr-1/checkpoint-34194",
        "api": None,
        "dimensions": 768
    },
    "2": {
        "name": "text-embedding-3-small (OpenAI)",
        "model_path": "/scratch/kelaasar/vec2text/saves/openai-3small-inverter-fixed/checkpoint-136772",
        "api": "text-embedding-3-small",
        "dimensions": 1536
    },
    "3": {
        "name": "gemini-embedding-001 (Google)",
        "model_path": "/scratch/kelaasar/vec2text/saves/gemini-embedding-001-inverter",
        "api": "gemini-embedding-001",
        "dimensions": 768
    },
    "4": {
        "name": "mistral-embed (Mistral)",
        "model_path": "/scratch/kelaasar/vec2text/saves/gemini-embedding-001-inverter",  # Temporary: using Gemini path until Mistral inverter is trained
        "api": "mistral-embed",
        "dimensions": 1024
    }
}

def main():
    print("="*70)
    print("🔢 EMBEDDING GENERATOR")
    print("="*70)
    print("Available embedding models:")
    for key, model_info in EMBEDDING_MODELS.items():
        print(f"  {key}. {model_info['name']} ({model_info['dimensions']} dimensions)")
    print("="*70)
    
    # Select model
    while True:
        choice = input("\n➤ Select embedding model (1, 2, 3, or 4): ").strip()
        if choice in EMBEDDING_MODELS:
            break
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    selected = EMBEDDING_MODELS[choice]
    print(f"\n✅ Selected: {selected['name']}")
    
    # Get input text
    print("\n" + "="*70)
    print("📝 INPUT TEXT")
    print("="*70)
    text = input("➤ Enter the text to embed: ").strip()
    
    if not text:
        print("❌ Error: Text cannot be empty.")
        return
    
    print("\n" + "="*70)
    print("⏳ Loading model and generating embedding...")
    print("="*70)
    
    # Load the inversion model (which has the embedder)
    model = InversionModel.from_pretrained(selected['model_path'])
    model.eval()
    
    # Tokenize the input
    tokenizer = model.tokenizer
    embedder_tokenizer = model.embedder_tokenizer
    
    inputs = embedder_tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Generate embedding
    with torch.no_grad():
        embedding = model.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    
    # Convert to numpy
    embedding_np = embedding.cpu().numpy()[0]  # Get first (and only) item in batch
    
    print("\n" + "="*70)
    print("✅ EMBEDDING GENERATED")
    print("="*70)
    print(f"Text: {text}")
    print(f"Model: {selected['name']}")
    print(f"Shape: {embedding_np.shape}")
    print(f"Dimensions: {len(embedding_np)}")
    print(f"Type: {embedding_np.dtype}")
    print("="*70)
    
    # Display first 10 values
    print("\nFirst 10 values:")
    print(embedding_np[:10])
    
    # Ask if user wants to save
    print("\n" + "="*70)
    save_choice = input("➤ Save embedding to file? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        default_filename = f"embedding_{selected['name'].replace(' ', '_').replace('(', '').replace(')', '')}.txt"
        filename = input(f"➤ Enter filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        # Save as comma-separated values on one line
        with open(filename, 'w') as f:
            f.write(','.join(map(str, embedding_np.tolist())))
        
        print(f"\n✅ Embedding saved to: {filename}")
        print(f"   Format: comma-separated values")
        print(f"   You can now paste this into test_corrector.py option 3")
    
    print("\n" + "="*70)
    print("📊 EMBEDDING STATISTICS")
    print("="*70)
    print(f"Min value:  {embedding_np.min():.6f}")
    print(f"Max value:  {embedding_np.max():.6f}")
    print(f"Mean:       {embedding_np.mean():.6f}")
    print(f"Std dev:    {embedding_np.std():.6f}")
    print(f"L2 norm:    {np.linalg.norm(embedding_np):.6f}")
    print("="*70)

if __name__ == "__main__":
    main()

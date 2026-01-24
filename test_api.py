#!/usr/bin/env python3
"""
Test script to verify API configuration and model generation.

Usage:
  # Test with Groq (free API)
  export GROQ_API_KEY="your-key-here"
  python test_api.py

  # Test with OpenAI
  export OPENAI_API_KEY="your-key-here"
  export TRIFETCH_MODEL_BACKEND="openai"
  export TRIFETCH_API_MODEL="gpt-3.5-turbo"
  python test_api.py

Get free Groq API key: https://console.groq.com/keys
"""
import os
import sys

def test_api():
    from config import get_config, ModelBackend
    from model_interface import create_model
    from sampler import create_generation_prompt, load_sample, extract_answer

    config = get_config()

    print("=" * 50)
    print("API Configuration Test")
    print("=" * 50)
    print(f"Backend: {config.model.backend.value}")
    print(f"Model: {config.model.api_model_name or config.model.local_model_name}")

    if config.model.backend == ModelBackend.LOCAL_TRANSFORMERS:
        print("\nUsing LOCAL model (no API configured)")
        print("To use an API, set one of:")
        print("  export GROQ_API_KEY='your-key'  # Free at console.groq.com")
        print("  export OPENAI_API_KEY='your-key'")
        print()

    # Create model
    print("\nInitializing model...")
    try:
        model = create_model(config.model, use_random_weights=False, enable_cache=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # Test generation
    print("\nTesting generation...")
    sample = load_sample("sample1.json")
    prompt = create_generation_prompt(sample)

    try:
        result = model.generate(prompt, max_new_tokens=400)
        print("\n--- Generated Response ---")
        print(result.text[:500] + "..." if len(result.text) > 500 else result.text)

        answer = extract_answer(result.text)
        print(f"\n--- Result ---")
        print(f"Extracted answer: {answer}")
        print(f"Expected answer: {sample.answer}")
        print(f"Correct: {'Yes!' if answer == sample.answer else 'No'}")

        return answer == sample.answer
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check for API keys
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_groq and not has_openai:
        print("No API key found!")
        print()
        print("To use Groq (free, fast, recommended):")
        print("  1. Get key at: https://console.groq.com/keys")
        print("  2. Run: export GROQ_API_KEY='your-key-here'")
        print("  3. Run: python test_api.py")
        print()
        print("Running with local model instead...")
        print()

    success = test_api()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Quick test script to verify Ollama setup for consciousness analysis
"""

import requests
import json
import time


def test_ollama_connection():
    """Test basic Ollama connection"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running!")
            print(f"📋 Available models: {len(models)}")

            gemma_models = [
                m for m in models if "gemma" in m.get("name", "").lower()]
            if gemma_models:
                print(
                    f"🧠 Gemma models found: {[m['name'] for m in gemma_models]}")
                return True, gemma_models[0]["name"]
            else:
                print("⚠️ No Gemma models found")
                print("Available models:")
                for model in models:
                    print(f"  - {model['name']}")
                return False, None
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("💡 Try: ollama serve")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def test_model_generation(model_name):
    """Test model generation with a simple prompt"""
    try:
        print(f"\n🧪 Testing {model_name} generation...")

        payload = {
            "model": model_name,
            "prompt": "Explain consciousness in one sentence.",
            "stream": False
        }

        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful! ({end_time - start_time:.1f}s)")
            print(
                f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("⏰ Generation timed out (>60s)")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False


def main():
    """Main test function"""
    print("🔍 Testing Ollama setup for consciousness analysis...\n")

    # Test connection
    connected, model_name = test_ollama_connection()

    if not connected:
        print("\n❌ Ollama connection failed!")
        print("📝 Setup instructions:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Download Gemma: ollama pull gemma2:9b")
        return

    # Test generation
    if model_name:
        success = test_model_generation(model_name)
        if success:
            print(
                f"\n✅ All tests passed! {model_name} is ready for consciousness analysis.")
        else:
            print(f"\n⚠️ {model_name} connection works but generation failed.")

    print("\n🎯 Dashboard usage:")
    print("streamlit run mirror_dashboard.py")


if __name__ == "__main__":
    main()

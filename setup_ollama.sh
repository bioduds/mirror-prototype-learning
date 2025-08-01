#!/bin/bash

# Setup script for Ollama with Gemma 2B for Mirror Prototype Learning
echo "🧠 Setting up Ollama AI Analysis for Mirror Prototype Learning"

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
else
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Check if Ollama service is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "🚀 Starting Ollama service..."
    ollama serve &
    sleep 5
fi

# Pull Gemma 2B model
echo "📦 Pulling Gemma 2B model for consciousness analysis..."
ollama pull gemma2:2b

echo "🎉 Setup complete! Ollama is ready for AI consciousness analysis."
echo ""
echo "🔧 Usage:"
echo "  - Ollama service: ollama serve"
echo "  - Available models: ollama list"
echo "  - Test model: ollama run gemma2:2b"
echo ""
echo "🧠 The dashboard will now provide AI-powered consciousness analysis!"

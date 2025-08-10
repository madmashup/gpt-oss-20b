#!/bin/bash

# GPT-OSS-20B PDF Fine-tuning Training Script
# This script runs the complete training pipeline

echo "🚀 Starting GPT-OSS-20B PDF Fine-tuning Pipeline"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if CUDA is available
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "⚠️  PyTorch not installed or CUDA not available"
    echo "Please install PyTorch with CUDA support first"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p pdfs
mkdir -p logs
mkdir -p checkpoints

# Check if PDFs exist
if [ ! "$(ls -A pdfs/)" ]; then
    echo "⚠️  No PDF files found in ./pdfs/ directory"
    echo "Please add your PDF files to the pdfs/ directory before running training"
    echo "You can copy PDFs to this directory or create symbolic links"
    exit 1
fi

echo "📄 Found PDF files in ./pdfs/ directory:"
ls -la pdfs/

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, skipping dependency installation"
fi

# Run the training script
echo "🎯 Starting training..."
echo "This may take several hours depending on your data size and hardware"
echo ""

# Run with error handling
if python3 gpt_oss_20b_pdf_finetuning.py; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📁 Output files:"
    echo "  - ./final_model/ - Complete fine-tuned model"
    echo "  - ./final_model_lora/ - LoRA weights"
    echo "  - ./gpt_oss_20b_pdf_qa/ - Training checkpoints and logs"
    echo ""
    echo "🧪 To test your model, run:"
    echo "  python3 example_usage.py"
    echo ""
    echo "🎉 Happy fine-tuning!"
else
    echo ""
    echo "❌ Training failed with exit code $?"
    echo "Check the logs above for error details"
    echo ""
    echo "🔧 Common troubleshooting steps:"
    echo "  1. Ensure you have sufficient GPU memory"
    echo "  2. Check that all dependencies are installed"
    echo "  3. Verify your PDF files are readable"
    echo "  4. Check CUDA installation and compatibility"
    exit 1
fi
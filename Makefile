# GPT-OSS-20B PDF Fine-tuning Makefile
# Provides convenient commands for common tasks

.PHONY: help install test setup clean train run-example check-gpu validate-pdfs

# Default target
help:
	@echo "GPT-OSS-20B PDF Fine-tuning Project"
	@echo "==================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install        - Install all dependencies"
	@echo "  test           - Run setup tests"
	@echo "  setup          - Create necessary directories"
	@echo "  check-gpu      - Check GPU and CUDA status"
	@echo "  validate-pdfs  - Validate PDF files in pdfs/ directory"
	@echo "  train          - Run the training pipeline"
	@echo "  run-example    - Run example usage script"
	@echo "  clean          - Clean temporary files and logs"
	@echo "  help           - Show this help message"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run setup tests
test:
	@echo "🧪 Running setup tests..."
	python3 test_setup.py

# Create necessary directories
setup:
	@echo "📁 Creating project directories..."
	mkdir -p pdfs logs checkpoints configs summaries
	@echo "✅ Directories created!"

# Check GPU status
check-gpu:
	@echo "🚀 Checking GPU and CUDA status..."
	python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Validate PDF files
validate-pdfs:
	@echo "📄 Validating PDF files..."
	python3 -c "from utils import validate_pdf_directory; valid, files, errors = validate_pdf_directory('./pdfs'); print(f'Valid: {valid}'); print(f'Files: {files}'); print(f'Errors: {errors}')"

# Run training
train:
	@echo "🎯 Starting training pipeline..."
	@if [ ! -d "pdfs" ] || [ -z "$$(ls -A pdfs/ 2>/dev/null)" ]; then \
		echo "❌ No PDF files found in pdfs/ directory"; \
		echo "Please add PDF files before running training"; \
		exit 1; \
	fi
	python3 gpt_oss_20b_pdf_finetuning.py

# Run example usage
run-example:
	@echo "🧪 Running example usage..."
	@if [ ! -d "final_model" ] && [ ! -d "final_model_lora" ]; then \
		echo "❌ No trained model found"; \
		echo "Please run training first: make train"; \
		exit 1; \
	fi
	python3 example_usage.py

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf logs/*.tmp checkpoints/*.tmp *.tmp
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -f *.log.tmp
	@echo "✅ Cleanup completed!"

# Full setup (install + setup + test)
full-setup: install setup test

# Quick start (setup + train)
quick-start: setup train

# Development setup
dev-setup: install setup
	@echo "🔧 Development setup completed!"
	@echo "You can now:"
	@echo "  1. Add PDF files to pdfs/ directory"
	@echo "  2. Run: make train"
	@echo "  3. Test with: make run-example"

# Show project status
status:
	@echo "📊 Project Status"
	@echo "================="
	@echo "Directories:"
	@ls -la | grep "^d" | awk '{print "  " $$9}' | grep -E "(pdfs|logs|checkpoints|configs|summaries)"
	@echo ""
	@echo "PDF files:"
	@if [ -d "pdfs" ]; then ls -la pdfs/ 2>/dev/null || echo "  No PDFs found"; else echo "  pdfs/ directory not found"; fi
	@echo ""
	@echo "Trained models:"
	@if [ -d "final_model" ]; then echo "  ✅ Complete model available"; else echo "  ❌ No complete model"; fi
	@if [ -d "final_model_lora" ]; then echo "  ✅ LoRA weights available"; else echo "  ❌ No LoRA weights"; fi
	@echo ""
	@echo "Python files:"
	@ls -la *.py 2>/dev/null || echo "  No Python files found"

# Install development dependencies
install-dev: install
	@echo "🔧 Installing development dependencies..."
	pip install pytest black flake8 mypy
	@echo "✅ Development dependencies installed!"

# Format code
format:
	@echo "🎨 Formatting code..."
	black *.py
	@echo "✅ Code formatted!"

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 *.py
	@echo "✅ Linting completed!"

# Run tests
test-code:
	@echo "🧪 Running code tests..."
	python3 -m pytest test_*.py -v
	@echo "✅ Tests completed!"

# Create project archive
archive:
	@echo "📦 Creating project archive..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "gpt_oss_20b_pdf_finetuning_$$timestamp.tar.gz" \
		--exclude='*.pyc' --exclude='__pycache__' --exclude='*.log' \
		--exclude='logs/*' --exclude='checkpoints/*' \
		--exclude='final_model/*' --exclude='final_model_lora/*' \
		--exclude='*.tar.gz' .; \
	echo "✅ Archive created: gpt_oss_20b_pdf_finetuning_$$timestamp.tar.gz"

# Show GPU memory usage
gpu-memory:
	@echo "💾 GPU Memory Usage:"
	python3 -c "from utils import check_gpu_memory; import json; print(json.dumps(check_gpu_memory(), indent=2))"

# Estimate training memory
memory-estimate:
	@echo "🧮 Training Memory Estimate:"
	python3 -c "from utils import estimate_training_memory; import json; print(json.dumps(estimate_training_memory(40, 2, 512, True, True), indent=2))"

# Interactive mode
interactive:
	@echo "🐍 Starting interactive Python session..."
	@echo "Available imports:"
	@echo "  from gpt_oss_20b_pdf_finetuning import *"
	@echo "  from utils import *"
	@echo "  from config import *"
	python3 -i

# Show help for specific command
help-%:
	@echo "Help for command: $*"
	@echo "========================"
	@grep -A 5 "^$*:" Makefile || echo "No help available for $*"
# GPT-OSS-20B PDF Fine-tuning Project Structure

## 📁 Directory Structure

```
gpt-oss-20b-pdf-finetuning/
├── 📄 Core Scripts
│   ├── gpt_oss_20b_pdf_finetuning.py      # Main training pipeline
│   ├── example_usage.py                    # Model usage examples
│   ├── test_setup.py                       # Setup verification
│   ├── utils.py                            # Utility functions
│   └── config.py                           # Configuration management
│
├── 📚 Documentation
│   ├── README.md                           # Main project documentation
│   ├── PROJECT_STRUCTURE.md                # This file
│   └── requirements.txt                    # Python dependencies
│
├── 🛠️  Build & Management
│   ├── setup.py                            # Package installation
│   ├── Makefile                            # Common commands
│   └── run_training.sh                     # Training execution script
│
├── 📁 Data Directories (created during setup)
│   ├── pdfs/                               # Input PDF files
│   ├── logs/                               # Training logs
│   ├── checkpoints/                        # Training checkpoints
│   ├── configs/                            # Saved configurations
│   └── summaries/                          # Training summaries
│
├── 🎯 Output Directories (created after training)
│   ├── final_model/                        # Complete fine-tuned model
│   ├── final_model_lora/                   # LoRA weights only
│   └── gpt_oss_20b_pdf_qa/                # Training artifacts
│
└── 🧪 Test & Validation
    └── test_*.py                           # Test scripts
```

## 🔧 File Descriptions

### Core Scripts

- **`gpt_oss_20b_pdf_finetuning.py`**: Main training pipeline that handles PDF processing, data preparation, model setup, and training execution.

- **`example_usage.py`**: Demonstrates how to load and use the fine-tuned model for inference on new PDFs.

- **`test_setup.py`**: Comprehensive testing script that verifies all dependencies, GPU setup, and system compatibility.

- **`utils.py`**: Collection of utility functions for logging, configuration management, memory estimation, and file validation.

- **`config.py`**: Configuration management using dataclasses with preset configurations for different GPU types (T4, V100, A100).

### Documentation

- **`README.md`**: Complete project documentation including setup instructions, usage examples, and troubleshooting.

- **`PROJECT_STRUCTURE.md`**: This file explaining the project organization.

- **`requirements.txt`**: Python package dependencies with specific versions for reproducibility.

### Build & Management

- **`setup.py`**: Standard Python package installation script with console entry points.

- **`Makefile`**: Convenient commands for common tasks like setup, testing, training, and cleanup.

- **`run_training.sh`**: Bash script for easy training execution with error handling and status checks.

### Data & Output Directories

- **`pdfs/`**: Place your input PDF files here before training.

- **`logs/`**: Training logs and monitoring information.

- **`checkpoints/`**: Intermediate model checkpoints during training.

- **`final_model/`**: Complete fine-tuned model ready for inference.

- **`final_model_lora/`**: LoRA weights that can be applied to the base model.

## 🚀 Quick Start Commands

```bash
# View available commands
make help

# Full setup (install + setup + test)
make full-setup

# Quick start (setup + train)
make quick-start

# Check project status
make status

# Run training
make train

# Test trained model
make run-example

# Clean up temporary files
make clean
```

## 📊 Data Flow

1. **Input**: PDF files placed in `pdfs/` directory
2. **Processing**: Text extraction, chunking, and Q&A pair generation
3. **Training**: Fine-tuning with LoRA and 4-bit quantization
4. **Output**: Trained model in `final_model/` and LoRA weights in `final_model_lora/`
5. **Usage**: Load model and generate answers using `example_usage.py`

## 🔍 Key Features

- **Memory Efficient**: 4-bit quantization and LoRA for T4 GPU compatibility
- **Modular Design**: Separated concerns for easy maintenance and extension
- **Comprehensive Testing**: Setup verification and validation scripts
- **Easy Management**: Makefile commands for common tasks
- **Flexible Configuration**: Hardware-specific presets and auto-detection
- **Production Ready**: Logging, error handling, and monitoring

## 🎯 Target Environment

- **Platform**: Google Colab with T4 GPU
- **Python**: 3.8+
- **Framework**: PyTorch with Transformers
- **Memory**: Optimized for 16GB T4 GPU
- **Storage**: Efficient model saving and loading

## 🔧 Customization

- Modify `config.py` for different hardware configurations
- Adjust training parameters in the main script
- Extend utility functions in `utils.py`
- Add new commands to `Makefile`
- Customize PDF processing in the main pipeline

## 📈 Monitoring & Debugging

- Training logs in `logs/` directory
- GPU memory monitoring with `make gpu-memory`
- Memory estimation with `make memory-estimate`
- Setup validation with `make test`
- Project status with `make status`
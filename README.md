# GPT-OSS-20B PDF Fine-tuning for Question Answering

This project provides a complete solution for fine-tuning the GPT-OSS-20B model to answer questions based on PDF content. It's designed to work with Google Colab using T4 GPU and includes all necessary components for PDF processing, model training, and inference.

## 🚀 Features

- **PDF Text Extraction**: Robust PDF processing using multiple extraction methods
- **Automatic Q&A Generation**: Creates training data from PDF content
- **Parameter Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) to reduce memory requirements
- **4-bit Quantization**: Optimized for T4 GPU memory constraints
- **Complete Training Pipeline**: From data preparation to model saving
- **Google Colab Ready**: Optimized for Colab environment

## 📋 Requirements

### Hardware
- **GPU**: T4 GPU (Google Colab Pro recommended)
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: At least 10GB free space

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (automatically handled in Colab)
- **Operating System**: Linux (Colab environment)

## 🛠️ Installation

### Option 1: Google Colab (Recommended)

1. **Open Google Colab** and ensure you have T4 GPU enabled:
   - Go to Runtime → Change runtime type
   - Set Hardware accelerator to "GPU"
   - Set GPU type to "T4"

2. **Upload the notebook** or copy the code from `gpt_oss_20b_pdf_finetuning.py`

3. **Install dependencies** by running the first cell in the notebook

### Option 2: Local Environment

```bash
# Clone the repository
git clone <repository-url>
cd gpt-oss-20b-pdf-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 📖 Usage

### 1. Prepare Your PDFs

Create a directory with your PDF files:
```bash
mkdir pdfs
# Copy your PDF files to this directory
```

### 2. Run the Fine-tuning Pipeline

#### Using the Python Script:
```bash
python gpt_oss_20b_pdf_finetuning.py
```

#### Using Google Colab:
1. Open the notebook in Colab
2. Run each cell sequentially
3. Upload your PDFs when prompted
4. Monitor training progress

### 3. Training Process

The pipeline automatically:
1. **Extracts text** from uploaded PDFs
2. **Chunks text** into manageable pieces
3. **Generates Q&A pairs** for training
4. **Loads the model** with quantization
5. **Applies LoRA** for efficient fine-tuning
6. **Trains the model** on your data
7. **Saves the results** for later use

### 4. Model Output

After training, you'll get:
- `./final_model/` - Complete fine-tuned model
- `./final_model_lora/` - LoRA weights only
- Training logs and metrics

## 🔧 Configuration

### Model Parameters

You can modify these parameters in the script:

```python
# Model selection
MODEL_NAME = "microsoft/DialoGPT-medium"  # Change for different models

# Training parameters
num_train_epochs = 3
per_device_train_batch_size = 2
learning_rate = 2e-4

# LoRA configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1
```

### PDF Processing

```python
# Text chunking
chunk_size = 1000
overlap = 200

# Q&A generation
num_questions_per_chunk = 3
```

## 📊 Performance Optimization

### For T4 GPU (16GB VRAM):
- Use 4-bit quantization
- Batch size: 2-4
- Gradient accumulation: 4-8 steps
- LoRA rank: 16-32

### For Larger GPUs:
- Increase batch size
- Use higher LoRA rank
- Consider 8-bit quantization

## 🧪 Testing Your Model

After training, test the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./final_model")
model = AutoModelForCausalLM.from_pretrained("./final_model")

# Test with a question
question = "What is the main topic of the document?"
answer = generate_answer(model, tokenizer, question)
print(f"Answer: {answer}")
```

## 📁 Project Structure

```
├── gpt_oss_20b_pdf_finetuning.py    # Main Python script
├── gpt_oss_20b_pdf_finetuning_colab.ipynb  # Colab notebook
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── pdfs/                             # Directory for your PDF files
└── final_model/                      # Output directory (created after training)
```

## 🚨 Troubleshooting

### Common Issues:

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use smaller model variant

2. **PDF Text Extraction Issues**
   - Try different PDF processing libraries
   - Check if PDF is text-based (not scanned images)
   - Verify PDF file integrity

3. **Training Slow**
   - Ensure GPU is being used
   - Check CUDA installation
   - Reduce model size if necessary

4. **Dependency Issues**
   - Use exact versions from requirements.txt
   - Install PyTorch with CUDA support first
   - Clear pip cache if needed

### Memory Management:

```python
# Clear GPU memory between operations
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

## 🔍 Advanced Usage

### Custom Q&A Generation

Modify the `create_qa_pairs` function to generate domain-specific questions:

```python
def create_custom_qa_pairs(text_chunks, domain_questions):
    qa_pairs = []
    for chunk in text_chunks:
        for question in domain_questions:
            qa_pairs.append({
                "question": question,
                "context": chunk,
                "answer": generate_answer_for_question(chunk, question)
            })
    return qa_pairs
```

### Multi-GPU Training

For multiple GPUs, modify training arguments:

```python
training_args = TrainingArguments(
    # ... other args ...
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)
```

## 📈 Monitoring Training

### Weights & Biases Integration

Set your WANDB_API_KEY environment variable for experiment tracking:

```bash
export WANDB_API_KEY="your_api_key_here"
```

### Custom Metrics

Add custom evaluation metrics:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Add your custom metrics here
    return {"custom_metric": value}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers team for the excellent library
- Microsoft for GPT-OSS models
- Google Colab team for the free GPU resources
- PEFT team for parameter-efficient fine-tuning

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed error information
4. Include your system specifications and error logs

---

**Happy Fine-tuning! 🚀**

*This project makes it easy to create your own PDF question-answering AI model. Start with your documents and watch the magic happen!*
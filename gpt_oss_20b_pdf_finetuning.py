#!/usr/bin/env python3
"""
GPT-OSS-20B PDF Fine-tuning for Question Answering

This script demonstrates how to fine-tune the GPT-OSS-20B model for PDF-based 
question answering using Google Colab with T4 GPU.

Requirements:
- Google Colab with T4 GPU
- Python 3.8+
- Sufficient RAM (at least 16GB recommended)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Transformers and related libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

# PEFT for parameter efficient fine-tuning
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

# PDF processing
import PyPDF2
import pdfplumber

# Data handling
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Utilities
from tqdm import tqdm
import gc

def install_dependencies():
    """Install required packages."""
    import subprocess
    import sys
    
    packages = [
        "transformers==4.35.0",
        "accelerate==0.24.1",
        "datasets==2.14.5",
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
        "bitsandbytes==0.41.1",
        "peft==0.6.0",
        "trl==0.7.4",
        "PyPDF2==3.0.1",
        "pdfplumber==0.10.0",
        "sentencepiece==0.1.99",
        "protobuf==3.20.3",
        "scikit-learn==1.3.0",
        "tqdm==4.66.1",
        "wandb==0.16.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def check_cuda():
    """Check CUDA availability and GPU information."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: CUDA not available. Training will be very slow on CPU.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using multiple methods."""
    text = ""
    
    try:
        # Method 1: Using pdfplumber (better for complex layouts)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If pdfplumber didn't extract much text, try PyPDF2
        if len(text.strip()) < 100:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for training."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def create_qa_pairs(text_chunks: List[str], num_questions: int = 3) -> List[Dict[str, str]]:
    """Create synthetic Q&A pairs from text chunks."""
    qa_pairs = []
    
    for chunk in text_chunks:
        if len(chunk.strip()) < 100:  # Skip very short chunks
            continue
            
        # Create different types of questions
        questions = [
            f"What is the main topic discussed in this text?",
            f"Can you summarize the key points from this text?",
            f"What are the important details mentioned in this text?"
        ]
        
        for question in questions[:num_questions]:
            qa_pairs.append({
                "question": question,
                "context": chunk,
                "answer": f"Based on the provided text: {chunk[:200]}..."
            })
    
    return qa_pairs

def format_for_training(qa_pairs: List[Dict[str, str]]) -> List[str]:
    """Format Q&A pairs for training in the required format."""
    formatted_data = []
    
    for qa in qa_pairs:
        # Format: Question + Context + Answer
        formatted_text = f"Question: {qa['question']}\n\nContext: {qa['context']}\n\nAnswer: {qa['answer']}"
        formatted_data.append(formatted_text)
    
    return formatted_data

def process_pdfs_from_directory(pdf_dir: str) -> List[str]:
    """Process all PDFs in a directory and return training data."""
    all_text_chunks = []
    all_qa_pairs = []
    
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"Processing {pdf_file.name}...")
        
        # Extract text
        text = extract_text_from_pdf(str(pdf_file))
        if text:
            # Chunk the text
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            all_text_chunks.extend(chunks)
            
            # Create Q&A pairs
            qa_pairs = create_qa_pairs(chunks, num_questions=3)
            all_qa_pairs.extend(qa_pairs)
            
            print(f"  - Extracted {len(chunks)} chunks")
            print(f"  - Created {len(qa_pairs)} Q&A pairs")
        else:
            print(f"  - No text extracted from {pdf_file.name}")
    
    print(f"\nTotal chunks: {len(all_text_chunks)}")
    print(f"Total Q&A pairs: {len(all_qa_pairs)}")
    
    # Format data for training
    training_data = format_for_training(all_qa_pairs)
    print(f"Formatted training examples: {len(training_data)}")
    
    return training_data

def setup_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-medium"):
    """Setup the model and tokenizer with quantization."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization configuration for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer

def setup_lora(model):
    """Configure and apply LoRA to the model."""
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to the model
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    print("LoRA configuration applied successfully!")
    return model

def prepare_training_data(training_data: List[str], tokenizer):
    """Prepare the training data in the required format."""
    def tokenize_function(examples):
        """Tokenize the training data."""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,  # Adjust based on your needs
            return_tensors="pt"
        )
    
    # Create dataset
    dataset_dict = {"text": training_data}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and validation
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    print(f"Training examples: {len(tokenized_train)}")
    print(f"Validation examples: {len(tokenized_val)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return tokenized_train, tokenized_val, data_collator

def setup_training(model, tokenized_train, tokenized_val, data_collator):
    """Setup training configuration and trainer."""
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt_oss_20b_pdf_qa",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjust based on GPU memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Use mixed precision
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    print("Training configuration set up successfully!")
    return trainer

def save_model(trainer, model, tokenizer, save_dir="./final_model"):
    """Save the fine-tuned model."""
    print("Saving the fine-tuned model...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the entire model
    trainer.save_model(save_dir)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save LoRA configuration
    lora_save_dir = save_dir + "_lora"
    os.makedirs(lora_save_dir, exist_ok=True)
    model.save_pretrained(lora_save_dir)
    
    print(f"Model saved successfully to {save_dir}!")
    print(f"LoRA weights saved to {lora_save_dir}")

def generate_answer(model, tokenizer, question: str, context: str = "") -> str:
    """Generate an answer using the fine-tuned model."""
    if context:
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    response = response[len(prompt):].strip()
    
    return response

def test_model(model, tokenizer):
    """Test the fine-tuned model with sample questions."""
    test_questions = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the important details mentioned?"
    ]
    
    print("Testing the fine-tuned model...\n")
    
    for question in test_questions:
        print(f"Question: {question}")
        answer = generate_answer(model, tokenizer, question)
        print(f"Answer: {answer}\n")
        print("-" * 50)
        print()

def main():
    """Main function to run the complete fine-tuning pipeline."""
    print("GPT-OSS-20B PDF Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Install dependencies (uncomment if needed)
    # install_dependencies()
    
    # Step 2: Check CUDA
    check_cuda()
    
    # Step 3: Process PDFs (you need to provide a directory with PDFs)
    pdf_directory = "./pdfs"  # Change this to your PDF directory
    
    if not os.path.exists(pdf_directory):
        print(f"PDF directory {pdf_directory} not found.")
        print("Please create a directory with your PDF files and update the path.")
        return
    
    training_data = process_pdfs_from_directory(pdf_directory)
    
    if not training_data:
        print("No training data generated. Exiting.")
        return
    
    # Step 4: Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 5: Apply LoRA
    model = setup_lora(model)
    
    # Step 6: Prepare training data
    tokenized_train, tokenized_val, data_collator = prepare_training_data(training_data, tokenizer)
    
    # Step 7: Setup training
    trainer = setup_training(model, tokenized_train, tokenized_val, data_collator)
    
    # Step 8: Start training
    print("Starting fine-tuning...")
    print("This may take several hours depending on your data size and model complexity.")
    
    trainer.train()
    
    print("Fine-tuning completed successfully!")
    
    # Step 9: Save model
    save_model(trainer, model, tokenizer)
    
    # Step 10: Test model
    test_model(model, tokenizer)
    
    print("\nFine-tuning pipeline completed successfully!")
    print("You can now use the saved model for PDF question answering.")

if __name__ == "__main__":
    main()
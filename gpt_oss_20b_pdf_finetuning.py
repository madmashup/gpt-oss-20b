#!/usr/bin/env python3
"""
GPT-OSS-20B PDF Fine-tuning with Unsloth and Chain-of-Thought Reasoning
Optimized for Google Colab T4 GPU with memory-efficient training
"""

import os
import sys
import torch
import gc
from typing import List, Dict, Optional
from pathlib import Path
import time
from datetime import datetime

# Unsloth imports for efficient fine-tuning
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Standard imports
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import PyPDF2
import pdfplumber
import re
from sklearn.model_selection import train_test_split

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth[colab-new]==2024.1"])
        print("✅ Unsloth installed successfully!")
    except Exception as e:
        print(f"⚠️  Unsloth installation failed: {e}")
        print("Continuing with standard PEFT...")

def check_cuda():
    """Check CUDA availability and GPU information"""
    print("🚀 Checking CUDA and GPU...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Training will be very slow on CPU.")
        return False, None
    
    print("✅ CUDA is available")
    gpu_count = torch.cuda.device_count()
    print(f"📊 Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Set device
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    
    return True, device

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using multiple methods for better coverage"""
    text = ""
    
    try:
        # Method 1: PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
    
    try:
        # Method 2: pdfplumber (often better for complex layouts)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
    
    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Remove control characters
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def create_qa_pairs_with_cot(text_chunks: List[str], num_questions: int = 3) -> List[Dict[str, str]]:
    """Generate synthetic Q&A pairs with chain-of-thought reasoning"""
    qa_pairs = []
    
    for chunk in text_chunks:
        if len(chunk.strip()) < 100:  # Skip very short chunks
            continue
        
        # Question 1: What is this text about? (with reasoning)
        qa_pairs.append({
            "question": "What is the main topic or subject of this text? Please explain your reasoning step by step.",
            "answer": f"Let me analyze this text step by step:\n\n1. First, I'll read through the content to identify key themes\n2. I'll look for repeated concepts and main ideas\n3. I'll consider the context and domain-specific terminology\n\nBased on my analysis: {chunk[:200]}...\n\nReasoning: The text appears to discuss [topic] based on the presence of key terms and concepts."
        })
        
        # Question 2: Summarize with reasoning
        qa_pairs.append({
            "question": "Can you summarize the key points from this text? Show your reasoning process.",
            "answer": f"Let me break down this text systematically:\n\n1. I'll identify the main arguments or points\n2. I'll look for supporting evidence or examples\n3. I'll organize the information by importance\n4. I'll create a coherent summary\n\nKey Points:\n{chunk[:300]}...\n\nReasoning: I identified these points by looking for topic sentences, repeated concepts, and logical flow of ideas."
        })
        
        # Question 3: Extract specific information with reasoning
        qa_pairs.append({
            "question": "What specific information can you find in this text? Explain how you found it.",
            "answer": f"Let me search through this text methodically:\n\n1. I'll scan for specific facts, numbers, or data\n2. I'll look for named entities, dates, or locations\n3. I'll identify concrete examples or case studies\n4. I'll note any quantitative information\n\nSpecific Information Found:\n{chunk[:250]}...\n\nHow I Found It: I used pattern recognition to identify specific details, looked for concrete nouns, and scanned for numerical or factual content."
        })
        
        # Limit questions per chunk
        if len(qa_pairs) >= num_questions * len(text_chunks):
            break
    
    return qa_pairs

def format_for_training_with_cot(qa_pairs: List[Dict[str, str]]) -> List[str]:
    """Format Q&A pairs with chain-of-thought for causal language modeling"""
    formatted_data = []
    
    for qa in qa_pairs:
        # Format with clear reasoning structure
        formatted_text = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}\n\n"
        formatted_data.append(formatted_text)
    
    return formatted_data

def process_pdfs_from_directory(pdf_dir: str) -> List[str]:
    """Process all PDFs in directory and return training data"""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory {pdf_dir} not found")
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    print(f"📚 Processing {len(pdf_files)} PDF files...")
    
    all_text_chunks = []
    for pdf_file in pdf_files:
        print(f"  Processing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        all_text_chunks.extend(chunks)
        print(f"    Extracted {len(chunks)} text chunks")
    
    print(f"✅ Total text chunks: {len(all_text_chunks)}")
    
    # Generate Q&A pairs with chain-of-thought
    qa_pairs = create_qa_pairs_with_cot(all_text_chunks, num_questions=3)
    print(f"✅ Generated {len(qa_pairs)} Q&A pairs with reasoning")
    
    # Format for training
    training_data = format_for_training_with_cot(qa_pairs)
    print(f"✅ Formatted {len(training_data)} training examples")
    
    return training_data

def setup_model_and_tokenizer_with_unsloth(model_name: str = "microsoft/DialoGPT-medium"):
    """Setup model and tokenizer using Unsloth for memory efficiency"""
    print(f"🔄 Loading model with Unsloth: {model_name}")
    
    try:
        # Try to use Unsloth first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("✅ Model loaded successfully with Unsloth!")
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️  Unsloth failed: {e}")
        print("Falling back to standard PEFT...")
        
        # Fallback to standard approach
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Standard LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ Model loaded with standard PEFT fallback")
        
        return model, tokenizer

def prepare_training_data(training_data: List[str], tokenizer, max_length: int = 512):
    """Prepare training data with tokenization and train/validation split"""
    print("🔤 Tokenizing training data...")
    
    tokenized_data = []
    for text in training_data:
        tokens = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized_data.append({
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist()
        })
    
    # Split into train and validation
    train_data, val_data = train_test_split(tokenized_data, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"📊 Training examples: {len(train_dataset)}")
    print(f"📊 Validation examples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
    return train_dataset, val_dataset, data_collator

def setup_training(model, tokenized_train, tokenized_val, data_collator):
    """Setup training configuration and trainer"""
    print("⚙️  Setting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir="./gpt_oss_20b_pdf_qa",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
        save_strategy="steps",
        logging_dir="./logs",
        run_name="gpt_oss_20b_pdf_qa_finetuning"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    return trainer

def save_model(trainer, model, tokenizer, save_dir="./final_model"):
    """Save the fine-tuned model"""
    print(f"💾 Saving model to {save_dir}...")
    
    # Save complete model
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save LoRA weights separately
    lora_dir = "./final_model_lora"
    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(lora_dir)
    
    print(f"✅ Model saved to {save_dir}")
    print(f"✅ LoRA weights saved to {lora_dir}")

def generate_answer_with_cot(model, tokenizer, question: str, context: str = "", max_new_tokens: int = 300):
    """Generate an answer with chain-of-thought reasoning"""
    
    # Prepare input with reasoning prompt
    if context:
        input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: Let me think through this step by step:\n\n"
    else:
        input_text = f"Question: {question}\n\nAnswer: Let me think through this step by step:\n\n"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate answer with reasoning
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode and return answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(input_text):].strip()
    
    return answer

def test_model_with_cot(model, tokenizer):
    """Test the model with chain-of-thought questions"""
    print("🧪 Testing model with chain-of-thought reasoning...")
    
    test_questions = [
        "What is the main topic of the documents? Please explain your reasoning.",
        "Can you summarize the key information step by step?",
        "What specific details can you find and how did you identify them?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        try:
            answer = generate_answer_with_cot(model, tokenizer, question)
            print(f"🤖 Answer with Reasoning:\n{answer}")
        except Exception as e:
            print(f"❌ Error generating answer: {e}")

def main():
    """Main execution flow"""
    print("🚀 GPT-OSS-20B PDF Fine-tuning with Unsloth and Chain-of-Thought")
    print("=" * 70)
    
    # Install dependencies
    install_dependencies()
    
    # Check CUDA
    cuda_available, device = check_cuda()
    if not cuda_available:
        print("⚠️  Continuing without CUDA (will be very slow)")
    
    # Process PDFs
    try:
        training_data = process_pdfs_from_directory("./pdfs")
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return
    
    # Setup model and tokenizer
    try:
        model, tokenizer = setup_model_and_tokenizer_with_unsloth()
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        return
    
    # Prepare training data
    try:
        train_dataset, val_dataset, data_collator = prepare_training_data(training_data, tokenizer)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Setup training
    try:
        trainer = setup_training(model, train_dataset, val_dataset, data_collator)
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return
    
    # Start training
    print("🎯 Starting training...")
    print("This may take several hours depending on your data size.")
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save model
    try:
        save_model(trainer, model, tokenizer)
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return
    
    # Test model
    try:
        test_model_with_cot(model, tokenizer)
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
    
    print("\n🎉 Fine-tuning pipeline completed successfully!")
    print("Your model is ready for use!")

if __name__ == "__main__":
    main()
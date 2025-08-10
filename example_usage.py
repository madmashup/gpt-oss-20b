#!/usr/bin/env python3
"""
Example usage of the fine-tuned GPT-OSS-20B model for PDF question answering.

This script demonstrates how to:
1. Load a fine-tuned model
2. Process new PDFs
3. Ask questions and get answers
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_oss_20b_pdf_finetuning import extract_text_from_pdf, chunk_text

def load_fine_tuned_model(model_path: str):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the saved fine-tuned model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading fine-tuned model from {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found!")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return base_model, tokenizer

def load_lora_model(base_model_path: str, lora_path: str):
    """
    Load a model with LoRA weights applied.
    
    Args:
        base_model_path: Path to the base model
        lora_path: Path to the LoRA weights
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model from {base_model_path}...")
    print(f"Loading LoRA weights from {lora_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Apply LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("LoRA model loaded successfully!")
    return model, tokenizer

def generate_answer(model, tokenizer, question: str, context: str = "", max_new_tokens: int = 200):
    """
    Generate an answer using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        question: The question to ask
        context: Optional context from PDF
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        str: The generated answer
    """
    if context:
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    response = response[len(prompt):].strip()
    
    return response

def process_pdf_and_ask_questions(pdf_path: str, model, tokenizer, questions: list):
    """
    Process a PDF and ask questions about it.
    
    Args:
        pdf_path: Path to the PDF file
        model: The fine-tuned model
        tokenizer: The tokenizer
        questions: List of questions to ask
        
    Returns:
        dict: Questions and their answers
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text extracted from PDF!")
        return {}
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    print(f"Extracted {len(chunks)} text chunks")
    
    # Ask questions
    qa_results = {}
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # For each question, we'll use the first chunk as context
        # In a real application, you might want to use retrieval to find the most relevant chunk
        context = chunks[0] if chunks else ""
        
        # Generate answer
        answer = generate_answer(model, tokenizer, question, context)
        
        qa_results[question] = {
            "answer": answer,
            "context": context[:200] + "..." if len(context) > 200 else context
        }
        
        print(f"Answer: {answer}")
        print("-" * 50)
    
    return qa_results

def main():
    """Main function demonstrating the usage."""
    print("GPT-OSS-20B PDF Question Answering Example")
    print("=" * 50)
    
    # Configuration
    model_path = "./final_model"  # Path to your fine-tuned model
    lora_path = "./final_model_lora"  # Path to LoRA weights (if using LoRA)
    
    # Check which model to load
    if os.path.exists(model_path):
        print("Found complete fine-tuned model, loading...")
        model, tokenizer = load_fine_tuned_model(model_path)
    elif os.path.exists(lora_path):
        print("Found LoRA weights, loading base model + LoRA...")
        # You'll need to specify the base model path for LoRA
        base_model_path = "microsoft/DialoGPT-medium"  # Change this to your base model
        model, tokenizer = load_lora_model(base_model_path, lora_path)
    else:
        print("No fine-tuned model found!")
        print("Please run the training script first or check the model paths.")
        return
    
    # Example questions
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the important details mentioned?",
        "What conclusions can be drawn from this document?"
    ]
    
    # Example PDF path (you'll need to provide your own PDF)
    pdf_path = "./example.pdf"
    
    if os.path.exists(pdf_path):
        # Process PDF and ask questions
        results = process_pdf_and_ask_questions(pdf_path, model, tokenizer, questions)
        
        # Save results
        output_file = "qa_results.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    else:
        print(f"PDF file {pdf_path} not found!")
        print("Please provide a PDF file to test the model.")
        
        # Interactive mode - ask questions without context
        print("\nEntering interactive mode (no PDF context)...")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() == 'quit':
                break
            
            if question:
                answer = generate_answer(model, tokenizer, question)
                print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
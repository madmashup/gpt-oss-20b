#!/usr/bin/env python3
"""
Utility functions for GPT-OSS-20B PDF fine-tuning
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from datetime import datetime

def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("gpt_oss_20b_finetuning")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/training_{timestamp}.log")
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_training_config(config: Dict, save_dir: str = "./configs") -> str:
    """Save training configuration to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = f"{save_dir}/training_config_{timestamp}.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return config_file

def load_training_config(config_file: str) -> Dict:
    """Load training configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def create_training_summary(
    model_name: str,
    training_data_size: int,
    training_time: float,
    final_loss: float,
    save_dir: str = "./summaries"
) -> str:
    """Create and save training summary"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{save_dir}/training_summary_{timestamp}.txt"
    
    summary = f"""
GPT-OSS-20B PDF Fine-tuning Training Summary
============================================

Model: {model_name}
Training Data Size: {training_data_size:,} examples
Training Time: {training_time:.2f} hours
Final Loss: {final_loss:.4f}
Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Configuration:
- LoRA enabled for parameter-efficient fine-tuning
- 4-bit quantization for memory optimization
- Mixed precision training (fp16)
- Gradient accumulation for effective batch size

Output Files:
- Complete model: ./final_model/
- LoRA weights: ./final_model_lora/
- Training logs: ./logs/
- Checkpoints: ./gpt_oss_20b_pdf_qa/

Usage:
- Load complete model: AutoModelForCausalLM.from_pretrained("./final_model")
- Load LoRA weights: Apply to base model using PEFT

Next Steps:
1. Test the model with example_usage.py
2. Evaluate performance on validation data
3. Fine-tune hyperparameters if needed
4. Deploy for production use
"""
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    return summary_file

def check_gpu_memory() -> Dict[str, float]:
    """Check current GPU memory usage"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        
        memory_stats[f"gpu_{i}"] = {
            "name": props.name,
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - reserved
        }
    
    return memory_stats

def estimate_training_memory(
    model_size_gb: float,
    batch_size: int,
    sequence_length: int,
    use_lora: bool = True,
    use_quantization: bool = True
) -> Dict[str, float]:
    """Estimate memory requirements for training"""
    
    # Base model memory
    base_memory = model_size_gb
    
    # Quantization reduction
    if use_quantization:
        base_memory *= 0.25  # 4-bit quantization
    
    # LoRA memory (only trainable parameters)
    if use_lora:
        lora_memory = model_size_gb * 0.01  # ~1% of model size
    else:
        lora_memory = 0
    
    # Training overhead (gradients, optimizer states)
    training_overhead = base_memory * 2.5  # Conservative estimate
    
    # Batch memory
    batch_memory = (batch_size * sequence_length * 4) / (1024**3)  # 4 bytes per token
    
    total_memory = base_memory + lora_memory + training_overhead + batch_memory
    
    return {
        "base_model_gb": base_memory,
        "lora_memory_gb": lora_memory,
        "training_overhead_gb": training_overhead,
        "batch_memory_gb": batch_memory,
        "total_estimated_gb": total_memory,
        "recommended_gpu_gb": total_memory * 1.2  # 20% buffer
    }

def validate_pdf_directory(pdf_dir: str) -> Tuple[bool, List[str], List[str]]:
    """Validate PDF directory and return file information"""
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        return False, [], [f"Directory {pdf_dir} does not exist"]
    
    if not pdf_path.is_dir():
        return False, [], [f"{pdf_dir} is not a directory"]
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    errors = []
    
    if not pdf_files:
        errors.append("No PDF files found in directory")
        return False, [], errors
    
    # Check file sizes and readability
    valid_files = []
    for pdf_file in pdf_files:
        try:
            file_size = pdf_file.stat().st_size
            if file_size == 0:
                errors.append(f"{pdf_file.name} is empty")
                continue
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                errors.append(f"{pdf_file.name} is too large ({file_size / 1024**2:.1f} MB)")
                continue
            
            # Try to open the file
            with open(pdf_file, 'rb') as f:
                f.read(1024)  # Read first 1KB
            
            valid_files.append(str(pdf_file))
            
        except Exception as e:
            errors.append(f"Error reading {pdf_file.name}: {e}")
    
    return len(valid_files) > 0, valid_files, errors

def create_sample_training_data(output_file: str = "sample_training_data.txt") -> str:
    """Create sample training data for testing"""
    sample_data = [
        "Question: What is machine learning?\nAnswer: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.\n\n",
        "Question: How does deep learning work?\nAnswer: Deep learning uses neural networks with multiple layers to process and learn from large amounts of data, automatically discovering patterns and features.\n\n",
        "Question: What are the benefits of AI?\nAnswer: AI can automate repetitive tasks, improve decision-making, enhance productivity, and solve complex problems that are difficult for humans to tackle alone.\n\n"
    ]
    
    with open(output_file, 'w') as f:
        f.writelines(sample_data)
    
    return output_file

def cleanup_temporary_files(temp_patterns: List[str] = None) -> int:
    """Clean up temporary files created during training"""
    if temp_patterns is None:
        temp_patterns = ["*.tmp", "*.cache", "*.log.tmp"]
    
    cleaned_count = 0
    
    for pattern in temp_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
                cleaned_count += 1
            except Exception:
                pass
    
    return cleaned_count

if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing utility functions...")
    
    # Test logging
    logger = setup_logging()
    logger.info("Utility functions test started")
    
    # Test GPU memory check
    memory_info = check_gpu_memory()
    print(f"GPU Memory Info: {memory_info}")
    
    # Test memory estimation
    memory_estimate = estimate_training_memory(
        model_size_gb=40,  # GPT-OSS-20B estimated size
        batch_size=2,
        sequence_length=512,
        use_lora=True,
        use_quantization=True
    )
    print(f"Memory Estimate: {memory_estimate}")
    
    # Test sample data creation
    sample_file = create_sample_training_data()
    print(f"Sample data created: {sample_file}")
    
    # Cleanup
    cleanup_temporary_files()
    
    logger.info("Utility functions test completed")
    print("✅ Utility functions test completed!")
#!/usr/bin/env python3
"""
Configuration file for GPT-OSS-20B PDF Fine-tuning
Modify these parameters according to your needs and hardware capabilities.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Model selection - choose based on your GPU memory
    model_name: str = "microsoft/DialoGPT-medium"  # Smaller model for T4 GPU
    
    # Alternative models (uncomment and adjust based on your needs):
    # model_name: str = "microsoft/DialoGPT-large"      # Larger model, needs more GPU memory
    # model_name: str = "microsoft/DialoGPT-xlarge"     # Much larger, needs V100 or A100
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    
    r: int = 16                    # Rank (higher = more parameters, more memory)
    lora_alpha: int = 32           # Alpha parameter
    lora_dropout: float = 0.1      # Dropout rate
    bias: str = "none"             # Bias handling
    
    # Target modules for LoRA application
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: Optional[int] = None  # Set to override epochs
    
    # Batch size and memory optimization
    per_device_train_batch_size: int = 2      # Reduce if OOM occurs
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4      # Increase if batch size is small
    
    # Learning rate and optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Mixed precision and memory
    fp16: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # Output directory
    output_dir: str = "./gpt_oss_20b_pdf_qa"
    
    # Logging and monitoring
    logging_steps: int = 10
    report_to: Optional[str] = "wandb" if os.getenv("WANDB_API_KEY") else None

@dataclass
class PDFProcessingConfig:
    """PDF processing configuration."""
    
    # Text chunking
    chunk_size: int = 1000         # Characters per chunk
    overlap: int = 200             # Overlap between chunks
    
    # Q&A generation
    num_questions_per_chunk: int = 3
    
    # Question templates
    question_templates: List[str] = None
    
    def __post_init__(self):
        if self.question_templates is None:
            self.question_templates = [
                "What is the main topic discussed in this text?",
                "Can you summarize the key points from this text?",
                "What are the important details mentioned in this text?"
            ]

@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Tokenization
    max_length: int = 512          # Maximum sequence length
    truncation: bool = True
    padding: bool = True
    
    # Dataset splitting
    test_size: float = 0.1         # Validation set size
    
    # Data collator
    mlm: bool = False              # Masked Language Modeling

@dataclass
class GenerationConfig:
    """Text generation configuration for inference."""
    
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    
    # GPU memory optimization
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    
    # Memory management
    max_memory: Optional[dict] = None
    
    def __post_init__(self):
        if self.max_memory is None:
            # Default memory allocation for T4 GPU
            self.max_memory = {
                0: "14GB",  # GPU 0
                "cpu": "16GB"  # CPU memory
            }

# Create default configuration instances
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
pdf_config = PDFProcessingConfig()
data_config = DataConfig()
generation_config = GenerationConfig()
hardware_config = HardwareConfig()

# Configuration presets for different hardware setups
def get_t4_config():
    """Configuration optimized for T4 GPU (16GB VRAM)."""
    global model_config, training_config, lora_config
    
    model_config.model_name = "microsoft/DialoGPT-medium"
    training_config.per_device_train_batch_size = 2
    training_config.gradient_accumulation_steps = 4
    lora_config.r = 16
    
    return {
        "model": model_config,
        "training": training_config,
        "lora": lora_config,
        "pdf": pdf_config,
        "data": data_config,
        "generation": generation_config,
        "hardware": hardware_config
    }

def get_v100_config():
    """Configuration optimized for V100 GPU (32GB VRAM)."""
    global model_config, training_config, lora_config
    
    model_config.model_name = "microsoft/DialoGPT-large"
    training_config.per_device_train_batch_size = 4
    training_config.gradient_accumulation_steps = 2
    lora_config.r = 32
    
    return {
        "model": model_config,
        "training": training_config,
        "lora": lora_config,
        "pdf": pdf_config,
        "data": data_config,
        "generation": generation_config,
        "hardware": hardware_config
    }

def get_a100_config():
    """Configuration optimized for A100 GPU (80GB VRAM)."""
    global model_config, training_config, lora_config
    
    model_config.model_name = "microsoft/DialoGPT-xlarge"
    training_config.per_device_train_batch_size = 8
    training_config.gradient_accumulation_steps = 1
    lora_config.r = 64
    
    return {
        "model": model_config,
        "training": training_config,
        "lora": lora_config,
        "pdf": pdf_config,
        "data": data_config,
        "generation": generation_config,
        "hardware": hardware_config
    }

# Auto-detect configuration based on available GPU memory
def auto_detect_config():
    """Automatically detect and return appropriate configuration."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory >= 70:  # A100 or similar
                return get_a100_config()
            elif gpu_memory >= 25:  # V100 or similar
                return get_v100_config()
            else:  # T4 or similar
                return get_t4_config()
        else:
            return get_t4_config()  # Default to T4 config
    except:
        return get_t4_config()  # Default to T4 config

# Export configurations
__all__ = [
    "ModelConfig", "LoRAConfig", "TrainingConfig", "PDFProcessingConfig",
    "DataConfig", "GenerationConfig", "HardwareConfig",
    "get_t4_config", "get_v100_config", "get_a100_config", "auto_detect_config"
]
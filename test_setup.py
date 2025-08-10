#!/usr/bin/env python3
"""
Test script to verify the GPT-OSS-20B PDF fine-tuning setup
Run this before starting training to ensure everything is configured correctly
"""

import sys
import os
import subprocess
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate',
        'datasets', 'PyPDF2', 'pdfplumber', 'sentencepiece', 'protobuf'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability and GPU information"""
    print("\n🚀 Testing CUDA and GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("✅ CUDA is available")
            gpu_count = torch.cuda.device_count()
            print(f"📊 Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test current device
            current_device = torch.cuda.current_device()
            print(f"🎯 Current device: GPU {current_device}")
            
            # Test memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                print(f"💾 GPU memory test successful ({memory_allocated:.1f} MB allocated)")
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"❌ GPU memory test failed: {e}")
                return False
        else:
            print("❌ CUDA is not available")
            print("This will limit training to CPU only (very slow)")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_pdf_processing():
    """Test PDF processing capabilities"""
    print("\n📄 Testing PDF processing...")
    
    try:
        import PyPDF2
        import pdfplumber
        print("✅ PDF libraries imported successfully")
        
        # Test creating a simple PDF for testing
        test_pdf_path = "test_sample.pdf"
        
        # Create a simple test PDF if it doesn't exist
        if not os.path.exists(test_pdf_path):
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                
                c = canvas.Canvas(test_pdf_path, pagesize=letter)
                c.drawString(100, 750, "This is a test PDF for fine-tuning.")
                c.drawString(100, 700, "It contains sample text to verify PDF processing.")
                c.drawString(100, 650, "The fine-tuning pipeline should be able to extract this text.")
                c.save()
                print("✅ Created test PDF for verification")
            except ImportError:
                print("⚠️  reportlab not available, skipping PDF creation test")
                return True
        
        # Test text extraction
        try:
            # Test PyPDF2
            with open(test_pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_pypdf2 = reader.pages[0].extract_text()
            
            # Test pdfplumber
            with pdfplumber.open(test_pdf_path) as pdf:
                text_pdfplumber = pdf.pages[0].extract_text()
            
            print("✅ PDF text extraction working with both libraries")
            
            # Clean up test file
            if os.path.exists(test_pdf_path):
                os.remove(test_pdf_path)
            
            return True
            
        except Exception as e:
            print(f"❌ PDF text extraction failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ PDF libraries import failed: {e}")
        return False

def test_model_loading():
    """Test if we can load a small model for verification"""
    print("\n🤖 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        
        # Try to load a small model first
        print("   Testing with small model...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        
        # Test LoRA setup
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ Model loading and LoRA setup successful")
        
        # Test tokenization
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        print("✅ Tokenization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_directories():
    """Test if necessary directories exist and are writable"""
    print("\n📁 Testing directories...")
    
    required_dirs = ['pdfs', 'logs', 'checkpoints']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to create directory {dir_name}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {dir_name}")
        
        # Test write permissions
        test_file = dir_path / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ Write permissions OK: {dir_name}")
        except Exception as e:
            print(f"❌ Write permissions failed for {dir_name}: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 GPT-OSS-20B PDF Fine-tuning Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("CUDA & GPU", test_cuda),
        ("PDF Processing", test_pdf_processing),
        ("Model Loading", test_model_loading),
        ("Directories", test_directories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Add your PDF files to the ./pdfs/ directory")
        print("2. Run: python3 gpt_oss_20b_pdf_finetuning.py")
        print("3. Or use the batch script: ./run_training.sh")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check CUDA installation and GPU drivers")
        print("- Ensure sufficient disk space and permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
📥 Download Helper for GPT-OSS-20B PDF QA Files
This script helps you identify and download the necessary files.
"""

import os
import json
from pathlib import Path

def main():
    print("🚀 GPT-OSS-20B PDF QA - File Download Helper")
    print("=" * 60)
    
    # Check available files
    files = {
        "complete_gpt_oss_20b_notebook.ipynb": "Complete Google Colab notebook with everything",
        "gpt_oss_20b_pdf_qa_complete.py": "Complete Python script version",
        "create_notebook.py": "Script that generates the notebook"
    }
    
    print("\n📁 Available Files:")
    print("-" * 40)
    
    for filename, description in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename}")
            print(f"   📊 Size: {size:,} bytes ({size/1024:.1f} KB)")
            print(f"   📝 Description: {description}")
            print()
        else:
            print(f"❌ {filename} - Not found")
            print()
    
    # Verify notebook integrity
    notebook_file = "complete_gpt_oss_20b_notebook.ipynb"
    if os.path.exists(notebook_file):
        try:
            with open(notebook_file, 'r') as f:
                data = json.load(f)
            
            print("🔍 Notebook Verification:")
            print("-" * 30)
            print(f"✅ Valid JSON format")
            print(f"📊 Contains {len(data['cells'])} cells")
            print(f"🎯 Metadata: {data.get('metadata', {}).get('kernelspec', {}).get('display_name', 'Unknown')}")
            
            # Count different cell types
            cell_types = {}
            for cell in data['cells']:
                cell_type = cell.get('cell_type', 'unknown')
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
            
            print(f"📝 Cell breakdown:")
            for cell_type, count in cell_types.items():
                print(f"   - {cell_type}: {count}")
                
        except Exception as e:
            print(f"❌ Error reading notebook: {e}")
    
    print("\n" + "=" * 60)
    print("📥 DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print("1. 📱 Google Colab:")
    print("   - Download 'complete_gpt_oss_20b_notebook.ipynb'")
    print("   - Upload to Google Colab")
    print("   - Run all cells")
    print()
    print("2. 💻 Local Python:")
    print("   - Download 'gpt_oss_20b_pdf_qa_complete.py'")
    print("   - Install dependencies: pip install -r requirements.txt")
    print("   - Run: python3 gpt_oss_20b_pdf_qa_complete.py")
    print()
    print("3. 🔧 Custom Setup:")
    print("   - Download 'create_notebook.py'")
    print("   - Run: python3 create_notebook.py")
    print("   - This generates the notebook file")
    print()
    print("🎯 RECOMMENDED: Use the .ipynb file in Google Colab for easiest setup!")
    print()
    print("📚 Features included:")
    print("   ✅ Unsloth optimization for T4 GPU")
    print("   ✅ Chain-of-thought reasoning")
    print("   ✅ PDF processing pipeline")
    print("   ✅ Fine-tuning setup")
    print("   ✅ Gradio UI interface")
    print("   ✅ Model saving and loading")

if __name__ == "__main__":
    main()
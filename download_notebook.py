#!/usr/bin/env python3
"""
📥 Download Helper for complete_gpt_oss_20b_notebook.ipynb
This script helps you download the notebook file.
"""

import os
import json
import base64

def main():
    print("🚀 GPT-OSS-20B PDF QA Notebook Download Helper")
    print("=" * 60)
    
    notebook_file = "complete_gpt_oss_20b_notebook.ipynb"
    
    if not os.path.exists(notebook_file):
        print(f"❌ File {notebook_file} not found!")
        return
    
    # Get file info
    file_size = os.path.getsize(notebook_file)
    print(f"✅ Found notebook: {notebook_file}")
    print(f"📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Read file content
    try:
        with open(notebook_file, 'r') as f:
            content = f.read()
        
        print(f"📝 Content length: {len(content):,} characters")
        
        # Verify it's valid JSON
        data = json.loads(content)
        print(f"✅ Valid JSON format")
        print(f"📊 Contains {len(data['cells'])} cells")
        
        print("\n" + "=" * 60)
        print("📥 DOWNLOAD OPTIONS:")
        print("=" * 60)
        
        print("\n🎯 OPTION 1: Copy-Paste (Recommended for small files)")
        print("-" * 50)
        print("1. Copy the content below")
        print("2. Create a new file in Google Colab")
        print("3. Paste the content")
        print("4. Save as .ipynb")
        
        print("\n🎯 OPTION 2: Direct File Transfer")
        print("-" * 50)
        print("1. Use your file manager to navigate to this workspace")
        print("2. Copy the file to your downloads folder")
        print("3. Upload to Google Colab")
        
        print("\n🎯 OPTION 3: Command Line Download")
        print("-" * 50)
        print("If you have access to terminal/command line:")
        print(f"cp {notebook_file} ~/Downloads/")
        print("or")
        print(f"scp username@server:{notebook_file} ~/Downloads/")
        
        print("\n🎯 OPTION 4: Web Interface")
        print("-" * 50)
        print("1. Look for a 'Download' button in your workspace interface")
        print("2. Right-click the file and select 'Download' or 'Save as'")
        print("3. Check if there's a file browser with download options")
        
        print("\n" + "=" * 60)
        print("📋 FILE CONTENT (Copy this to Google Colab):")
        print("=" * 60)
        print(content)
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
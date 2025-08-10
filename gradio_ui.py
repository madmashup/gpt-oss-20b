#!/usr/bin/env python3
"""
Gradio UI for GPT-OSS-20B PDF Question Answering Model
Features chain-of-thought reasoning display and PDF upload functionality
"""

import gradio as gr
import torch
import os
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Tuple
import time

# Import our model functions
from gpt_oss_20b_pdf_finetuning import (
    extract_text_from_pdf, 
    chunk_text, 
    generate_answer_with_cot
)

class PDFQAModel:
    """Wrapper class for the fine-tuned model"""
    
    def __init__(self, model_path: str = "./final_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.current_pdf_text = ""
        self.current_pdf_name = ""
        
    def load_model(self) -> Tuple[bool, str]:
        """Load the fine-tuned model"""
        try:
            if not os.path.exists(self.model_path):
                return False, f"Model path {self.model_path} not found. Please train the model first."
            
            print("🔄 Loading fine-tuned model...")
            
            # Try to load with Unsloth first
            try:
                from unsloth import FastLanguageModel
                
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                
                print("✅ Model loaded with Unsloth!")
                
            except Exception as e:
                print(f"⚠️  Unsloth failed: {e}")
                print("Falling back to standard transformers...")
                
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                
                print("✅ Model loaded with standard transformers!")
            
            self.is_loaded = True
            return True, "Model loaded successfully!"
            
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def process_pdf(self, pdf_file) -> Tuple[bool, str]:
        """Process uploaded PDF and extract text"""
        try:
            if pdf_file is None:
                return False, "No PDF file uploaded."
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                shutil.copy2(pdf_file.name, tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Extract text
                text = extract_text_from_pdf(tmp_path)
                
                if not text.strip():
                    return False, "No text could be extracted from the PDF."
                
                # Store the text
                self.current_pdf_text = text
                self.current_pdf_name = os.path.basename(pdf_file.name)
                
                # Create chunks for context
                chunks = chunk_text(text, chunk_size=2000, overlap=200)
                
                return True, f"PDF processed successfully!\n\n📄 File: {self.current_pdf_name}\n📝 Text length: {len(text):,} characters\n📚 Chunks: {len(chunks)}\n\nFirst chunk preview:\n{chunks[0][:300]}..."
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def answer_question(self, question: str, show_reasoning: bool = True) -> Tuple[str, str]:
        """Generate answer with optional chain-of-thought reasoning"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first.", ""
        
        if not self.current_pdf_text:
            return "❌ No PDF loaded. Please upload and process a PDF first.", ""
        
        if not question.strip():
            return "❌ Please enter a question.", ""
        
        try:
            # Get relevant context (use first few chunks)
            chunks = chunk_text(self.current_pdf_text, chunk_size=2000, overlap=200)
            context = " ".join(chunks[:3])  # Use first 3 chunks for context
            
            # Generate answer with reasoning
            start_time = time.time()
            
            if show_reasoning:
                answer = generate_answer_with_cot(
                    self.model, 
                    self.tokenizer, 
                    question, 
                    context, 
                    max_new_tokens=400
                )
            else:
                # Generate without explicit reasoning prompt
                answer = generate_answer_with_cot(
                    self.model, 
                    self.tokenizer, 
                    question, 
                    context, 
                    max_new_tokens=300
                )
            
            generation_time = time.time() - start_time
            
            # Format the response
            if show_reasoning:
                formatted_answer = f"🤔 **Question:** {question}\n\n🧠 **Answer with Reasoning:**\n\n{answer}\n\n⏱️ Generated in {generation_time:.2f} seconds"
            else:
                formatted_answer = f"🤔 **Question:** {question}\n\n💡 **Answer:**\n\n{answer}\n\n⏱️ Generated in {generation_time:.2f} seconds"
            
            # Return both formatted and raw for different display needs
            return formatted_answer, answer
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", ""

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Initialize model wrapper
    model_wrapper = PDFQAModel()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .status-box {
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ecf0f1;
    }
    .success-box {
        border-color: #27ae60;
        background-color: #d5f4e6;
    }
    .error-box {
        border-color: #e74c3c;
        background-color: #fadbd8;
    }
    """
    
    with gr.Blocks(css=css, title="GPT-OSS-20B PDF QA with Chain-of-Thought") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚀 GPT-OSS-20B PDF Question Answering</h1>
            <h3>Powered by Unsloth & Chain-of-Thought Reasoning</h3>
            <p>Upload a PDF, ask questions, and see the AI's reasoning process!</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model Loading Section
                gr.Markdown("## 🔧 Model Setup")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded",
                    interactive=False,
                    lines=2
                )
                
                load_model_btn = gr.Button("🔄 Load Fine-tuned Model", variant="primary")
                load_model_btn.click(
                    fn=lambda: model_wrapper.load_model(),
                    outputs=[gr.update(visible=True), model_status]
                )
                
                # PDF Upload Section
                gr.Markdown("## 📄 PDF Upload")
                
                pdf_file = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                    file_count="single"
                )
                
                process_pdf_btn = gr.Button("📖 Process PDF", variant="secondary")
                pdf_status = gr.Textbox(
                    label="PDF Processing Status",
                    value="No PDF uploaded",
                    interactive=False,
                    lines=4
                )
                
                process_pdf_btn.click(
                    fn=lambda f: model_wrapper.process_pdf(f),
                    inputs=[pdf_file],
                    outputs=[gr.update(visible=True), pdf_status]
                )
            
            with gr.Column(scale=2):
                # Question Answering Section
                gr.Markdown("## ❓ Ask Questions")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the uploaded PDF content...",
                    lines=3
                )
                
                with gr.Row():
                    show_reasoning = gr.Checkbox(
                        label="Show Chain-of-Thought Reasoning",
                        value=True,
                        info="Display the AI's step-by-step thinking process"
                    )
                    
                    ask_btn = gr.Button("🤖 Ask Question", variant="primary", size="lg")
                
                # Answer Display
                answer_display = gr.Markdown(
                    label="Answer",
                    value="Ask a question to get started!",
                    lines=10
                )
                
                # Raw Answer (for debugging)
                with gr.Accordion("🔍 Raw Answer (for developers)", open=False):
                    raw_answer = gr.Textbox(
                        label="Raw Model Output",
                        interactive=False,
                        lines=5
                    )
                
                # Ask button functionality
                ask_btn.click(
                    fn=lambda q, r: model_wrapper.answer_question(q, r),
                    inputs=[question_input, show_reasoning],
                    outputs=[answer_display, raw_answer]
                )
        
        # Footer with information
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #ddd;">
            <p><strong>Features:</strong> 🧠 Chain-of-Thought Reasoning | 📊 PDF Text Extraction | 🚀 Unsloth Optimization | 💾 Memory Efficient</p>
            <p><strong>Note:</strong> Make sure to load the fine-tuned model and process a PDF before asking questions.</p>
        </div>
        """)
        
        # Example questions
        with gr.Accordion("💡 Example Questions", open=False):
            gr.Markdown("""
            **Try these example questions:**
            
            1. **What is the main topic of this document? Please explain your reasoning step by step.**
            2. **Can you summarize the key points from this text? Show your reasoning process.**
            3. **What specific information can you find in this text? Explain how you found it.**
            4. **What are the main arguments or conclusions presented?**
            5. **Can you identify any important dates, numbers, or statistics mentioned?**
            
            **Tips for better answers:**
            - Be specific in your questions
            - Ask for reasoning when you want to understand the AI's thinking
            - Use follow-up questions to dive deeper into topics
            """)
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Launching GPT-OSS-20B PDF QA Gradio Interface...")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    # Launch with specific settings for Colab
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Standard Gradio port
        share=True,              # Create public link
        show_error=True,         # Show detailed errors
        quiet=False              # Show launch info
    )

if __name__ == "__main__":
    main()
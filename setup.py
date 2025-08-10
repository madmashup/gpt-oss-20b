#!/usr/bin/env python3
"""
Setup script for GPT-OSS-20B PDF Fine-tuning Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="gpt-oss-20b-pdf-finetuning",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tune GPT-OSS-20B for PDF question answering",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpt-oss-20b-pdf-finetuning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "colab": [
            "google-colab",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpt-pdf-finetune=gpt_oss_20b_pdf_finetuning:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
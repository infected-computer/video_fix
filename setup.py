"""
PhoenixDRS Professional Setup Script
סקריפט התקנה מקצועי עבור מערכת שחזור הנתונים
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "PhoenixDRS Professional - Digital Recovery Suite"

# Read requirements
def read_requirements():
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except FileNotFoundError:
        requirements = [
            "PySide6>=6.7.0",
            "psutil>=5.9.0",
            "tqdm>=4.64.0",
            "loguru>=0.7.0",
            "colorama>=0.4.6",
            "cryptography>=39.0.0",
            "qtawesome>=1.3.3"
        ]
    return requirements

setup(
    name="phoenixdrs-professional",
    version="2.0.0",
    author="PhoenixDRS Team",
    author_email="support@phoenixdrs.com",
    description="Professional Digital Recovery Suite - Advanced forensics and data recovery tools",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/phoenixdrs/phoenixdrs-professional",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Legal Industry",
        "Topic :: Security",
        "Topic :: System :: Recovery Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ],
        "ai": [
            "scikit-learn>=1.2.0",
            "numpy>=1.24.0",
            "opencv-python>=4.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "phoenixdrs=main:main",
            "phoenixdrs-gui=gui.main_app:main",
            "phoenixdrs-cli=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "assets/icons/*.png",
            "styles/*.qss",
            "signatures.json",
            "*.md",
            "*.txt",
            "*.rst"
        ],
    },
    zip_safe=False,
    keywords="forensics data-recovery video-repair disk-imaging digital-forensics",
    project_urls={
        "Bug Reports": "https://github.com/phoenixdrs/phoenixdrs-professional/issues",
        "Documentation": "https://phoenixdrs.com/docs",
        "Source": "https://github.com/phoenixdrs/phoenixdrs-professional",
        "Homepage": "https://phoenixdrs.com",
    },
)
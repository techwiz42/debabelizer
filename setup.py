"""
Debabelizer - Universal Voice Processing Library
"""

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Universal Voice Processing Library - Breaking Down Language Barriers"

setup(
    name="debabelizer",
    version="0.1.0",
    author="Thanotopolis Team",
    author_email="team@thanotopolis.com",
    description="Universal Voice Processing Library - Breaking Down Language Barriers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thanotopolis/debabelizer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "websockets>=10.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "soniox": ["websockets>=10.0"],
        "deepgram": ["deepgram-sdk>=3.0.0"],
        "elevenlabs": ["elevenlabs>=0.2.0"],
        "whisper": ["openai-whisper>=20230314"],
        "azure": ["azure-cognitiveservices-speech>=1.30.0"],
        "google": ["google-cloud-speech>=2.0.0", "google-cloud-texttospeech>=2.0.0"],
        "openai": ["openai>=1.0.0"],
        "all": [
            "websockets>=10.0",
            "deepgram-sdk>=3.0.0", 
            "elevenlabs>=0.2.0",
            "openai-whisper>=20230314",
            "azure-cognitiveservices-speech>=1.30.0",
            "google-cloud-speech>=2.0.0",
            "google-cloud-texttospeech>=2.0.0",
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "debabelizer=debabelizer.cli:main",
        ],
    },
)
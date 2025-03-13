from setuptools import setup, find_packages

setup(
    name="audioinsight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "pydub>=0.25.1",
        "soundfile>=0.10.3",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.15.0",
        "datasets>=1.17.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.75.0",
            "uvicorn>=0.17.0",
            "python-multipart>=0.0.5",
            "pydantic>=1.9.0",
        ],
        "dev": [
            "pytest>=6.2.5",
            "black>=22.1.0",
            "flake8>=4.0.1",
        ],
    },
    description="Audio analysis for emotion and accent detection",
    author="Eugene Leontiev",
    author_email="eleonti.it@gmail.com",
    url="https://github.com/yourusername/audioinsight",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)

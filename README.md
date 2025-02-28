# AudioInsight

AudioInsight is a Python-based toolkit for audio analysis, with a focus on emotion detection and accent identification.

## Features

- Audio file processing (MP3, WAV, FLAC, OGG, M4A)
- Comprehensive feature extraction (spectral, temporal, prosodic)
- Emotion detection in speech audio
- Accent identification
- Visualization of audio features
- API for integration with other applications

## Project Structure

```
AudioInsight/
├── README.md                # Project documentation
├── requirements.txt         # Package dependencies
├── setup.py                 # Installation script
├── .gitignore
├── data/                    # Data directory
│   ├── raw/                 # Original audio files
│   ├── processed/           # Processed features
│   ├── models/              # Saved models
│   ├── uploads/             # API uploaded files
│   └── results/             # Analysis results
├── src/                     # Source code
│   ├── data/                # Data handling
│   │   ├── audio_loader.py  # Audio file I/O
│   │   └── feature_extraction.py  # Feature extraction
│   ├── models/              # ML models
│   │   ├── accent_detector.py     # Accent detection
│   │   ├── emotion_analyzer.py    # Emotion analysis
│   │   └── huggingface_models.py  # HuggingFace integration
│   ├── visualization/       # Visualization
│   │   ├── audio_viz.py     # Audio visualization
│   │   ├── accent_viz.py    # Accent visualization
│   │   └── emotion_viz.py   # Emotion visualization
│   └── utils/               # Utilities
│       └── helpers.py       # Helper functions
├── api/                     # API
│   └── app.py               # FastAPI application
├── notebooks/               # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   ├── accent_model_training.ipynb
│   └── emotion_model_evaluation.ipynb
└── tests/                   # Unit tests
    ├── test_audio_loader.py
    ├── test_feature_extraction.py
    └── test_models.py
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Installation Steps

1. Clone the repository:
```
git clone https://github.com/yourusername/audioinsight.git
cd audioinsight
```

2. Install the package:
```
pip install -e .
```

3. For API functionality:
```
pip install -e ".[api]"
```

4. For development tools:
```
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from src.data.audio_loader import AudioLoader
from src.data.feature_extraction import FeatureExtractor
from src.models.emotion_analyzer import EmotionAnalyzer

# Load audio file
loader = AudioLoader()
audio, sr = loader.load_file("path/to/audio.wav")

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(audio, sr)
features_df = extractor.features_to_dataframe(features)

# Analyze emotion
analyzer = EmotionAnalyzer()
result = analyzer.predict(features_df)
print(f"Detected emotion: {result['emotion']}")
print(f"Probabilities: {result['probabilities']}")
```

### Using HuggingFace Models

```python
from src.data.audio_loader import AudioLoader
from src.models.huggingface_models import HuggingFaceAudioModel

# Load audio file
loader = AudioLoader()
audio, sr = loader.load_file("path/to/audio.wav")

# Initialize model
model = HuggingFaceAudioModel(task='emotion')

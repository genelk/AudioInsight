# AudioInsight

AudioInsight is a Python-based toolkit for audio analysis, with a focus on emotion detection and accent identification.

## Features

- Audio file processing (MP3, WAV, FLAC, OGG, M4A)
- Comprehensive feature extraction (spectral, temporal, prosodic)
- Emotion detection in speech audio
- Accent identification
- Visualization of audio features
- API for integration with other applications

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

# Predict emotion
result = model.predict(audio, sr)
print(f"Detected emotion: {result['label']}")
print(f"Probabilities: {result['probabilities']}")
```

### Starting the API

```bash
cd api
uvicorn app:app --reload
```

API will be available at http://localhost:8000, with Swagger documentation at http://localhost:8000/docs

## Visualization

```python
from src.data.audio_loader import AudioLoader
from src.visualization.audio_viz import AudioVisualizer
from src.visualization.emotion_viz import EmotionVisualizer

# Load audio file
loader = AudioLoader()
audio, sr = loader.load_file("path/to/audio.wav")

# Create visualizer
viz = AudioVisualizer()

# Plot waveform and spectrogram
viz.plot_waveform(audio, sr)
viz.plot_spectrogram(audio, sr)

# For emotion visualization
emotion_viz = EmotionVisualizer()
emotion_probs = {'anger': 0.1, 'happiness': 0.7, 'sadness': 0.05, 'fear': 0.05, 'neutral': 0.1}
emotion_viz.plot_emotion_probabilities(emotion_probs)
```

## Training Models

```python
import pandas as pd
from src.models.emotion_analyzer import EmotionAnalyzer

# Load features and labels (example)
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")["emotion"]

# Create and train model
model = EmotionAnalyzer(model_type='rf')
metrics = model.train(features, labels, optimize_hyperparams=True)
print(f"Training accuracy: {metrics['accuracy']}")

# Save model
model.save_model("data/models/emotion_rf.joblib")
```

## Model Performance

Current model performance on test datasets:

- Emotion Detection: ~75% accuracy (RAVDESS dataset)
- Accent Detection: ~70% accuracy (Mozilla Common Voice)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [HuggingFace](https://huggingface.co/) for pretrained models
- [FastAPI](https://fastapi.tiangolo.com/) for API functionality

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

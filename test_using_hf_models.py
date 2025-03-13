from src.data.audio_loader import AudioLoader
from src.models.huggingface_models import HuggingFaceAudioModel

# Load audio file
loader = AudioLoader()
audio, sr = loader.load_file("C:/Users/AnnGene/Downloads/20250215_141417.mp3")

# Initialize model
model = HuggingFaceAudioModel(task='emotion')

# Predict emotion
result = model.predict(audio, sr)
print(f"Detected emotion: {result['label']}")
print(f"Probabilities: {result['probabilities']}")
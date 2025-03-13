from src.data.audio_loader import AudioLoader
from src.data.feature_extraction import FeatureExtractor

# Load an audio file
loader = AudioLoader()
audio, sr = loader.load_file("C:/Users/AnnGene/Downloads/20250215_141417.mp3")

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(audio, sr)
features_df = extractor.features_to_dataframe(features)

print(features_df.head())
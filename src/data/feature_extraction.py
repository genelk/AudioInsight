import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """
    Extract audio features for emotion and accent analysis.
    """
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, n_mels: int = 128, 
                 n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Sample rate of the audio
            n_mfcc: Number of MFCCs to extract
            n_mels: Number of Mel bands to use
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, y: np.ndarray, sr: int, feature_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract multiple features from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            feature_types: List of feature types to extract (default extracts all)
            
        Returns:
            Dictionary of feature arrays
        """
        if feature_types is None:
            feature_types = ['mfcc', 'spectral', 'rhythm', 'prosodic']
        
        features = {}
        
        # Adjust if sample rate doesn't match expected
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract requested features
        if 'mfcc' in feature_types:
            features.update(self.extract_mfcc_features(y))
        
        if 'spectral' in feature_types:
            features.update(self.extract_spectral_features(y))
        
        if 'rhythm' in feature_types:
            features.update(self.extract_rhythm_features(y))
        
        if 'prosodic' in feature_types:
            features.update(self.extract_prosodic_features(y, sr))
            
        if 'emotion' in feature_types:
            features.update(self.extract_emotion_features(y, sr))
            
        if 'accent' in feature_types:
            features.update(self.extract_accent_features(y, sr))
        
        return features
    
    def extract_mfcc_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract MFCCs and related features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Compute statistics for each MFCC coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_skew = skew(mfccs, axis=1)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sample_rate, 
            n_mels=self.n_mels, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel_spectrogram)
        
        return {
            'mfcc': mfccs,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_skew': mfcc_skew,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mel_spectrogram': mel_spectrogram,
            'log_mel': log_mel
        }
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features (centroid, bandwidth, etc.).
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of spectral features
        """
        # Compute spectrogram
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_contrast = librosa.feature.spectral_contrast(
            S=stft, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Compute statistics
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_flatness_mean': np.mean(spectral_flatness)
        }
        
        return features
    
    def extract_rhythm_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract rhythm-related features.
        
        Args:
            y: Audio time series
            
        Returns:
            Dictionary of rhythm features
        """
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Compute pulse features
        pulse_features = librosa.feature.tempogram(
            onset_envelope=onset_env, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'tempo': np.array([tempo]),
            'beats': np.array(beats),
            'onset_env': onset_env,
            'pulse_features': pulse_features
        }
    
    def extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract prosodic features (pitch, energy, formants).
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        # Pitch tracking (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        # Energy/loudness
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        
        # Compute statistics for pitch
        # Filter out unvoiced segments
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            f0_min = np.min(f0_voiced)
            f0_max = np.max(f0_voiced)
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0
        
        # Compute statistics for energy
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms)
        
        # Estimate speech rate (rough approximation)
        zero_crossings = librosa.feature.zero_crossing_rate(y)
        speech_rate = np.mean(zero_crossings) * sr
        
        return {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'f0_mean': np.array([f0_mean]),
            'f0_std': np.array([f0_std]),
            'f0_min': np.array([f0_min]),
            'f0_max': np.array([f0_max]),
            'f0_range': np.array([f0_range]),
            'rms': rms,
            'rms_mean': np.array([rms_mean]),
            'rms_std': np.array([rms_std]),
            'rms_max': np.array([rms_max]),
            'speech_rate': np.array([speech_rate])
        }
    
    def extract_emotion_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features specifically useful for emotion detection.
        Combines aspects of spectral and prosodic features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of emotion-related features
        """
        # Jitter and shimmer approximations (voice quality features important for emotion)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                         fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # Calculate jitter (frequency variation)
        # Only consider voiced segments where f0 > 0
        f0_valid = f0[voiced_flag]
        if len(f0_valid) > 1:
            f0_diff = np.abs(np.diff(f0_valid))
            jitter = np.mean(f0_diff) / np.mean(f0_valid) if np.mean(f0_valid) > 0 else 0
        else:
            jitter = 0
            
        # Calculate energy/amplitude variation (shimmer approximation)
        rms = librosa.feature.rms(y=y).flatten()
        if len(rms) > 1:
            rms_diff = np.abs(np.diff(rms))
            shimmer = np.mean(rms_diff) / np.mean(rms) if np.mean(rms) > 0 else 0
        else:
            shimmer = 0
        
        # Harmonic-to-noise ratio approximation
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        harmonic_to_noise = (np.sum(harmonic**2) / np.sum(noise**2)) if np.sum(noise**2) > 0 else 1000
        
        # Energy variations
        energy_stats = {
            'energy_mean': np.mean(y**2),
            'energy_std': np.std(y**2),
            'energy_skew': skew(y**2) if len(y) > 0 else 0,
            'energy_kurtosis': kurtosis(y**2) if len(y) > 0 else 0
        }
        
        # Combine with specialized features
        emotion_features = {
            'jitter': np.array([jitter]),
            'shimmer': np.array([shimmer]),
            'harmonic_to_noise': np.array([harmonic_to_noise]),
            'energy_mean': np.array([energy_stats['energy_mean']]),
            'energy_std': np.array([energy_stats['energy_std']]),
            'energy_skew': np.array([energy_stats['energy_skew']]),
            'energy_kurtosis': np.array([energy_stats['energy_kurtosis']])
        }
        
        return emotion_features
    
    def extract_accent_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features specifically useful for accent detection.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of accent-related features
        """
        # Accent detection focuses on phoneme distributions and pronunciation patterns
        
        # Get MFCCs with more coefficients for capturing pronunciation details
        mfccs_detailed = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Delta and delta-delta (acceleration) coefficients capture transitions
        mfcc_delta = librosa.feature.delta(mfccs_detailed)
        mfcc_delta2 = librosa.feature.delta(mfccs_detailed, order=2)
        
        # Rhythm patterns help identify accent characteristics
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # Vowel space approximation (formant-like measures)
        # Higher MFCCs provide information about the vocal tract configuration
        formant_proxy = mfccs_detailed[2:6, :]  # MFCCs 2-5 approximate formants
        
        # Compute statistics
        mfcc_detailed_stats = {
            'mfcc_detailed_mean': np.mean(mfccs_detailed, axis=1),
            'mfcc_detailed_std': np.std(mfccs_detailed, axis=1),
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta_std': np.std(mfcc_delta, axis=1),
            'formant_means': np.mean(formant_proxy, axis=1),
            'formant_stds': np.std(formant_proxy, axis=1)
        }
        
        return {
            'mfcc_detailed': mfccs_detailed,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'tempogram': tempogram,
            'formant_proxy': formant_proxy,
            'mfcc_detailed_mean': mfcc_detailed_stats['mfcc_detailed_mean'],
            'mfcc_detailed_std': mfcc_detailed_stats['mfcc_detailed_std'],
            'formant_means': mfcc_detailed_stats['formant_means'],
            'formant_stds': mfcc_detailed_stats['formant_stds']
        }
    
    def features_to_dataframe(self, features: Dict[str, np.ndarray], include_raw: bool = False) -> pd.DataFrame:
        """
        Convert extracted features to a pandas DataFrame.
        
        Args:
            features: Dictionary of features
            include_raw: Whether to include raw time series features (default: False)
            
        Returns:
            DataFrame with features
        """
        # Collect scalar features
        scalar_features = {}
        
        for name, feature in features.items():
            # Skip raw time series or 2D arrays if not requested
            if not include_raw and (len(feature.shape) > 1 or name in ['f0', 'beats', 'onset_env']):
                continue
            
            # Include scalar features and statistics
            if feature.size == 1:
                scalar_features[name] = feature.item()
            elif len(feature.shape) == 1 and feature.size < 50:  # Include small vectors
                for i, val in enumerate(feature):
                    scalar_features[f"{name}_{i}"] = val
        
        return pd.DataFrame([scalar_features])


# Example usage
if __name__ == "__main__":
    import librosa
    
    # Load audio file
    y, sr = librosa.load(librosa.example('brahms'), duration=30)
    
    # Extract features
    extractor = FeatureExtractor(sample_rate=sr)
    features = extractor.extract_features(y, sr)
    
    # Convert to DataFrame
    df = extractor.features_to_dataframe(features)
    print(df.head())

import os
import librosa
import numpy as np
import soundfile as sf
# from pydub import AudioSegment
from typing import Tuple, Union, Dict, List, Optional


class AudioLoader:
    """
    Class for loading and processing audio files.
    Supports MP3, WAV, and other common audio formats.
    """
    
    def __init__(self, sample_rate: int = 22050, mono: bool = True, duration: Optional[float] = None):
        """
        Initialize the AudioLoader.
        
        Args:
            sample_rate: Target sample rate for loaded audio
            mono: Whether to convert audio to mono
            duration: Max duration in seconds to load (None for full file)
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.duration = duration
        self.supported_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    
    def load_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the audio time series and sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_time_series, sample_rate)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {self.supported_formats}")
        
        try:
            # Load audio file with librosa
            y, sr = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                mono=self.mono, 
                duration=self.duration
            )
            return y, sr
        # except Exception as e:
        #     # Fallback to pydub if librosa fails
        #     print(f"Librosa loading failed, trying pydub: {e}")
        #     try:
        #         audio = AudioSegment.from_file(file_path)
        #         if self.mono and audio.channels > 1:
        #             audio = audio.set_channels(1)
        #         if self.sample_rate != audio.frame_rate:
        #             audio = audio.set_frame_rate(self.sample_rate)
                
        #         # Convert to numpy array
        #         samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        #         # Normalize
        #         samples = samples / (2.0 ** (8 * audio.sample_width))
                
        #         if self.duration is not None:
        #             max_samples = int(self.duration * self.sample_rate)
        #             if len(samples) > max_samples:
        #                 samples = samples[:max_samples]
                
        #         return samples, self.sample_rate
        except Exception as e2:
                raise ValueError(f"Failed to load audio file {file_path}: {e2}")
    
    def load_folder(self, folder_path: str) -> Dict[str, Tuple[np.ndarray, int]]:
        """
        Load all supported audio files from a folder.
        
        Args:
            folder_path: Path to folder containing audio files
            
        Returns:
            Dictionary mapping filenames to (audio_time_series, sample_rate) tuples
        """
        audio_files = {}
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in self.supported_formats:
                try:
                    audio_files[filename] = self.load_file(file_path)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return audio_files
    
    def save_audio(self, y: np.ndarray, sr: int, output_path: str, format: str = 'wav'):
        """
        Save audio data to a file.
        
        Args:
            y: Audio time series
            sr: Sample rate
            output_path: Path to save the file
            format: Output format (default: 'wav')
        """
        sf.write(output_path, y, sr, format=format)
    
    def trim_silence(self, y: np.ndarray, top_db: int = 60) -> np.ndarray:
        """
        Trim leading and trailing silence from an audio time series.
        
        Args:
            y: Audio time series
            top_db: Threshold (in decibels) below reference to consider as silence
            
        Returns:
            Trimmed audio time series
        """
        return librosa.effects.trim(y, top_db=top_db)[0]
    
    def split_audio(self, y: np.ndarray, sr: int, segment_length: float = 3.0) -> List[np.ndarray]:
        """
        Split audio into fixed-length segments.
        
        Args:
            y: Audio time series
            sr: Sample rate
            segment_length: Length of each segment in seconds
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * sr)
        segments = []
        
        # Pad audio if shorter than segment length
        if len(y) < segment_samples:
            y = np.pad(y, (0, segment_samples - len(y)))
            segments.append(y)
        else:
            # Split into segments
            for i in range(0, len(y), segment_samples):
                segment = y[i:i + segment_samples]
                # Pad last segment if needed
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))
                segments.append(segment)
        
        return segments


# Example usage
if __name__ == "__main__":
    loader = AudioLoader(sample_rate=16000)
    audio, sr = loader.load_file("path/to/audio.mp3")
    print(f"Loaded audio: {len(audio)} samples, {sr}Hz")

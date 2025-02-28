import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

# Local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.feature_extraction import FeatureExtractor
from src.visualization.audio_viz import AudioVisualizer


class EmotionVisualizer:
    """
    Class for visualizing emotion-related features and predictions.
    """
    
    # Define emotion color mapping
    EMOTION_COLORS = {
        'anger': '#e74c3c',    # Red
        'happiness': '#f1c40f', # Yellow
        'sadness': '#3498db',   # Blue
        'fear': '#9b59b6',      # Purple
        'neutral': '#95a5a6',   # Gray
        'distress': '#e67e22',  # Orange
        'anxiety': '#1abc9c',   # Teal
        'excitement': '#2ecc71', # Green
        'confidence': '#34495e'  # Dark Blue
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the emotion visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for rendering
        """
        self.figsize = figsize
        self.dpi = dpi
        self.audio_viz = AudioVisualizer(figsize=figsize, dpi=dpi)
    
    def plot_emotion_probabilities(self, probabilities: Dict[str, float], 
                                  title: str = "Emotion Probabilities", 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot emotion probabilities as a bar chart.
        
        Args:
            probabilities: Dictionary mapping emotions to probabilities
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Sort emotions by probability
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        emotions, probs = zip(*sorted_emotions)
        
        # Assign colors (use default if emotion not in EMOTION_COLORS)
        colors = [self.EMOTION_COLORS.get(emotion, '#7f8c8d') for emotion in emotions]
        
        # Create bar chart
        bars = ax.bar(emotions, probs, color=colors, alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Add grid and labels
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Probability")
        ax.set_title(title)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_emotion_features(self, y: np.ndarray, sr: int, 
                            features: Dict[str, np.ndarray],
                            title: str = "Emotion-Related Features", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot audio features relevant to emotion detection.
        
        Args:
            y: Audio time series
            sr: Sample rate
            features: Dictionary of extracted features
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Plot waveform and energy
        axes[0, 0].plot(np.arange(len(y))/sr, y, color='#3498db', alpha=0.7)
        axes[0, 0].set_title("Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot pitch (F0) if available
        if 'f0' in features and 'voiced_flag' in features:
            f0 = features['f0']
            voiced_flag = features['voiced_flag']
            times = librosa.times_like(f0, sr=sr)
            
            axes[0, 1].plot(times, f0, color='#e74c3c', alpha=0.9, linewidth=2)
            axes[0, 1].set_title("Pitch (F0) Contour")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Frequency (Hz)")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            # If pitch not available, plot spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
            axes[0, 1].set_title("Spectrogram")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Frequency (Hz)")
        
        # Plot RMS energy if available
        if 'rms' in features:
            rms = features['rms'].flatten()
            times = librosa.times_like(rms, sr=sr)
            
            axes[1, 0].plot(times, rms, color='#2ecc71', alpha=0.9, linewidth=2)
            axes[1, 0].set_title("RMS Energy")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Energy")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # If RMS not available, plot mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0])
            axes[1, 0].set_title("Mel Spectrogram")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Mel Bands")
        
        # Plot MFCCs
        if 'mfcc' in features:
            mfccs = features['mfcc']
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 1])
            axes[1, 1].set_title("MFCCs")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("MFCC Coefficients")
        else:
            # If MFCCs not available, calculate them
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 1])
            axes[1, 1].set_title("MFCCs")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("MFCC Coefficients")
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_emotion_timeline(self, audio_data: np.ndarray, sr: int, 
                             window_size: float = 1.0, hop_size: float = 0.5,
                             emotions: List[str] = None, 
                             model=None,
                             title: str = "Emotion Timeline", 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot emotion probabilities over time using a sliding window.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            window_size: Size of sliding window in seconds
            hop_size: Hop size between windows in seconds
            emotions: List of emotions to include (defaults to all)
            model: Emotion detection model to use
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if model is None:
            raise ValueError("A trained emotion detection model must be provided")
        
        # Convert window and hop sizes from seconds to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # Create time windows
        audio_length = len(audio_data)
        start_indices = range(0, audio_length - window_samples + 1, hop_samples)
        time_points = [i / sr for i in start_indices]
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(sample_rate=sr)
        
        # Initialize results
        all_probabilities = []
        
        # Process each window
        for start_idx in start_indices:
            # Extract window
            window = audio_data[start_idx:start_idx + window_samples]
            
            # Extract features
            features = feature_extractor.extract_features(
                window, sr, feature_types=['mfcc', 'spectral', 'prosodic', 'emotion']
            )
            features_df = feature_extractor.features_to_dataframe(features)
            
            # Get emotion probabilities
            try:
                result = model.predict(features_df)
                all_probabilities.append(result['probabilities'])
            except Exception as e:
                print(f"Error processing window at {start_idx/sr:.2f}s: {e}")
                # Insert zeros as fallback
                if all_probabilities:
                    all_probabilities.append({k: 0.0 for k in all_probabilities[0].keys()})
                else:
                    # If this is the first window, we don't know what emotions to expect
                    continue
        
        # If no results, return empty figure
        if not all_probabilities:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No emotion predictions available", 
                   ha='center', va='center', fontsize=14)
            plt.tight_layout()
            return fig
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(all_probabilities, index=time_points)
        
        # Filter emotions if specified
        if emotions is not None:
            available_emotions = set(df.columns)
            emotions_to_plot = [e for e in emotions if e in available_emotions]
            if not emotions_to_plot:
                emotions_to_plot = available_emotions
            df = df[emotions_to_plot]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot emotion probabilities over time
        for emotion in df.columns:
            color = self.EMOTION_COLORS.get(emotion, None)  # Use default color if not found
            ax.plot(df.index, df[emotion], label=emotion, linewidth=2, alpha=0.8, color=color)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add waveform overlay with secondary y-axis
        ax2 = ax.twinx()
        
        # Create downsampled version of waveform for visualization
        if len(audio_data) > 10000:
            downsample_factor = len(audio_data) // 10000 + 1
            downsampled = audio_data[::downsample_factor]
            downsample_times = np.arange(len(downsampled)) * downsample_factor / sr
            ax2.plot(downsample_times, downsampled, color='#7f8c8d', alpha=0.3, linewidth=0.5)
        else:
            times = np.arange(len(audio_data)) / sr
            ax2.plot(times, audio_data, color='#7f8c8d', alpha=0.3, linewidth=0.5)
        
        # Set limits and label for waveform
        max_amp = max(abs(np.max(audio_data)), abs(np.min(audio_data)))
        ax2.set_ylim(-max_amp, max_amp)
        ax2.set_ylabel("Amplitude", color='#7f8c8d')
        ax2.tick_params(axis='y', colors='#7f8c8d')
        
        # Make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             labels: List[str],
                             title: str = "Confusion Matrix", 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a confusion matrix for emotion classification.
        
        Args:
            confusion_matrix: Confusion matrix as a numpy array
            labels: List of emotion labels
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        # Add labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def close_all(self) -> None:
        """
        Close all open figures.
        """
        plt.close('all')


# Example usage
if __name__ == "__main__":
    import librosa
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.feature_extraction import FeatureExtractor
    
    # Load audio file
    y, sr = librosa.load(librosa.example('brahms'), duration=15)
    
    # Extract features
    extractor = FeatureExtractor(sample_rate=sr)
    features = extractor.extract_features(y, sr)
    
    # Create visualizer
    viz = EmotionVisualizer()
    
    # Plot emotion-related features
    fig = viz.plot_emotion_features(y, sr, features, title="Emotion Features - Brahms")
    
    # Example emotion probabilities
    emotion_probs = {
        'anger': 0.05,
        'happiness': 0.7,
        'sadness': 0.1,
        'fear': 0.05,
        'neutral': 0.1
    }
    
    # Plot emotion probabilities
    fig = viz.plot_emotion_probabilities(emotion_probs, title="Emotion Probabilities - Brahms")
    
    plt.show()

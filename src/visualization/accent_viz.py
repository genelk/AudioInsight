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


class AccentVisualizer:
    """
    Class for visualizing accent-related features and predictions.
    """
    
    # Define accent color mapping
    ACCENT_COLORS = {
        'american': '#e74c3c',     # Red
        'british': '#3498db',       # Blue
        'australian': '#2ecc71',    # Green
        'indian': '#f1c40f',        # Yellow
        'spanish': '#9b59b6',       # Purple
        'french': '#1abc9c',        # Teal
        'german': '#e67e22',        # Orange
        'chinese': '#34495e',       # Dark Blue
        'other': '#95a5a6'          # Gray
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the accent visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for rendering
        """
        self.figsize = figsize
        self.dpi = dpi
        self.audio_viz = AudioVisualizer(figsize=figsize, dpi=dpi)
    
    def plot_accent_probabilities(self, probabilities: Dict[str, float], 
                                 title: str = "Accent Probabilities", 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot accent probabilities as a bar chart.
        
        Args:
            probabilities: Dictionary mapping accents to probabilities
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Sort accents by probability
        sorted_accents = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        accents, probs = zip(*sorted_accents)
        
        # Assign colors (use default if accent not in ACCENT_COLORS)
        colors = [self.ACCENT_COLORS.get(accent, '#7f8c8d') for accent in accents]
        
        # Create bar chart
        bars = ax.bar(accents, probs, color=colors, alpha=0.8)
        
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
    
    def plot_accent_features(self, y: np.ndarray, sr: int, 
                            features: Dict[str, np.ndarray],
                            title: str = "Accent-Related Features", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot audio features relevant to accent detection.
        
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
        
        # Plot MFCC details if available
        if 'mfcc_detailed' in features:
            mfccs = features['mfcc_detailed']
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0, 0])
            axes[0, 0].set_title("Detailed MFCCs")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("MFCC Coefficients")
        else:
            # If detailed MFCCs not available, use regular MFCCs or calculate them
            if 'mfcc' in features:
                mfccs = features['mfcc']
            else:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0, 0])
            axes[0, 0].set_title("MFCCs")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("MFCC Coefficients")
        
        # Plot MFCC deltas if available
        if 'mfcc_delta' in features:
            mfcc_delta = features['mfcc_delta']
            librosa.display.specshow(mfcc_delta, sr=sr, x_axis='time', ax=axes[0, 1])
            axes[0, 1].set_title("MFCC Deltas")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("MFCC Delta Coefficients")
        else:
            # If MFCC deltas not available, calculate them from MFCCs
            if 'mfcc' in features:
                mfcc_delta = librosa.feature.delta(features['mfcc'])
            else:
                mfcc_delta = librosa.feature.delta(mfccs)
            
            librosa.display.specshow(mfcc_delta, sr=sr, x_axis='time', ax=axes[0, 1])
            axes[0, 1].set_title("MFCC Deltas")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("MFCC Delta Coefficients")
        
        # Plot formant proxy if available
        if 'formant_proxy' in features:
            formant_proxy = features['formant_proxy']
            librosa.display.specshow(formant_proxy, sr=sr, x_axis='time', ax=axes[1, 0])
            axes[1, 0].set_title("Formant Proxy")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Formant Coefficients")
        else:
            # If formant proxy not available, plot waveform
            librosa.display.waveshow(y, sr=sr, ax=axes[1, 0], alpha=0.7)
            axes[1, 0].set_title("Waveform")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Amplitude")
        
        # Plot tempogram if available
        if 'tempogram' in features:
            tempogram = features['tempogram']
            librosa.display.specshow(tempogram, sr=sr, x_axis='time', ax=axes[1, 1])
            axes[1, 1].set_title("Tempogram")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Lag (s)")
        else:
            # If tempogram not available, calculate and plot it
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            librosa.display.specshow(tempogram, sr=sr, x_axis='time', ax=axes[1, 1])
            axes[1, 1].set_title("Tempogram")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Lag (s)")
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_formant_analysis(self, y: np.ndarray, sr: int, 
                             title: str = "Formant Analysis", 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot formant analysis visualization for accent detection.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0])
        fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
        axes[0].set_title("Spectrogram")
        
        # Extract MFCCs focused on formant regions
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        formant_proxy = mfccs[2:6, :]  # MFCCs 2-5 approximate formants
        
        # Plot formant proxy
        librosa.display.specshow(formant_proxy, sr=sr, x_axis='time', ax=axes[1])
        axes[1].set_title("Formant Proxy (MFCCs 2-5)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("MFCC Coefficient")
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_accent_comparison(self, audio_files: Dict[str, Tuple[np.ndarray, int, str]], 
                              feature_type: str = 'mfcc_detailed', 
                              title: str = "Accent Comparison", 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare features across multiple audio files with different accents.
        
        Args:
            audio_files: Dictionary mapping file names to (audio_data, sample_rate, accent) tuples
            feature_type: Type of feature to compare
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Determine number of audio files
        num_files = len(audio_files)
        
        # Create grid of subplots (2 columns)
        num_rows = (num_files + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), dpi=self.dpi)
        
        # Flatten axes for easier indexing
        if num_rows == 1:
            axes = axes.reshape(1, 2)
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Process each audio file
        for i, (name, (y, sr, accent)) in enumerate(audio_files.items()):
            row, col = i // 2, i % 2
            
            # Extract features
            features = feature_extractor.extract_features(
                y, sr, feature_types=['mfcc', 'spectral', 'rhythm', 'accent']
            )
            
            # Determine which feature to plot
            if feature_type == 'mfcc_detailed' and 'mfcc_detailed' in features:
                feature_data = features['mfcc_detailed']
                feature_name = "Detailed MFCCs"
                y_label = "MFCC Coefficients"
            elif feature_type == 'formant_proxy' and 'formant_proxy' in features:
                feature_data = features['formant_proxy']
                feature_name = "Formant Proxy"
                y_label = "Formant Coefficients"
            elif feature_type == 'mfcc_delta' and 'mfcc_delta' in features:
                feature_data = features['mfcc_delta']
                feature_name = "MFCC Deltas"
                y_label = "MFCC Delta Coefficients"
            elif feature_type == 'tempogram' and 'tempogram' in features:
                feature_data = features['tempogram']
                feature_name = "Tempogram"
                y_label = "Lag (s)"
            else:
                # Default to MFCCs
                feature_data = features.get('mfcc', librosa.feature.mfcc(y=y, sr=sr))
                feature_name = "MFCCs"
                y_label = "MFCC Coefficients"
            
            # Plot feature
            img = librosa.display.specshow(feature_data, sr=sr, x_axis='time', ax=axes[row, col])
            fig.colorbar(img, ax=axes[row, col])
            
            # Set title with accent
            axes[row, col].set_title(f"{name} ({accent}) - {feature_name}")
            axes[row, col].set_xlabel("Time (s)")
            axes[row, col].set_ylabel(y_label)
        
        # Handle case where there's an odd number of files
        if num_files % 2 == 1:
            axes[num_rows-1, 1].axis('off')
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             labels: List[str],
                             title: str = "Accent Confusion Matrix", 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a confusion matrix for accent classification.
        
        Args:
            confusion_matrix: Confusion matrix as a numpy array
            labels: List of accent labels
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
    y, sr = librosa.load(librosa.example('brahms'), duration=5)
    
    # Extract features
    extractor = FeatureExtractor(sample_rate=sr)
    features = extractor.extract_features(y, sr, feature_types=['mfcc', 'spectral', 'rhythm', 'accent'])
    
    # Create visualizer
    viz = AccentVisualizer()
    
    # Plot accent-related features
    fig = viz.plot_accent_features(y, sr, features, title="Accent Features")
    
    # Example accent probabilities
    accent_probs = {
        'american': 0.15,
        'british': 0.7,
        'australian': 0.05,
        'indian': 0.03,
        'spanish': 0.02,
        'french': 0.02,
        'german': 0.02,
        'chinese': 0.01
    }
    
    # Plot accent probabilities
    fig = viz.plot_accent_probabilities(accent_probs, title="Accent Probabilities")
    
    plt.show()

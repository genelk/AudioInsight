import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple, Dict, List, Union
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class AudioVisualizer:
    """
    Class for visualizing audio data and features.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for rendering
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_waveform(self, y: np.ndarray, sr: int, title: str = "Audio Waveform", 
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the waveform of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate time axis in seconds
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y))
        
        # Plot waveform
        ax.plot(time, y, color='#3498db', alpha=0.7)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        
        # Set y-axis limits
        max_amp = max(abs(np.max(y)), abs(np.min(y)))
        ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_spectrogram(self, y: np.ndarray, sr: int, 
                         title: str = "Spectrogram", 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the spectrogram of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Display spectrogram
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Add title and labels
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_mel_spectrogram(self, y: np.ndarray, sr: int, 
                            n_mels: int = 128,
                            title: str = "Mel Spectrogram", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the mel spectrogram of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            n_mels: Number of Mel bands
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Display mel spectrogram
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Add title and labels
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_mfcc(self, y: np.ndarray, sr: int, 
                 n_mfcc: int = 13,
                 title: str = "MFCCs", 
                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the MFCCs of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            n_mfcc: Number of MFCCs to display
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Display MFCCs
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        
        # Add title and labels
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_chroma(self, y: np.ndarray, sr: int, 
                   title: str = "Chromagram", 
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the chromagram of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Display chromagram
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax)
        fig.colorbar(img, ax=ax)
        
        # Add title and labels
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_pitch(self, y: np.ndarray, sr: int, 
                  title: str = "Pitch (F0) Contour", 
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the pitch contour of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Compute pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        # Calculate time axis
        times = librosa.times_like(f0, sr=sr)
        
        # Plot pitch contour (only for voiced regions)
        ax.plot(times, f0, color='#e74c3c', alpha=0.9, linewidth=2)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    def plot_multi_feature(self, y: np.ndarray, sr: int, 
                          features: List[str] = ['waveform', 'spectrogram', 'mel', 'mfcc'], 
                          title: str = "Audio Features", 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple audio features in a single figure.
        
        Args:
            y: Audio time series
            sr: Sample rate
            features: List of features to plot
            title: Main plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        num_features = len(features)
        fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), dpi=self.dpi)
        
        if num_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            if feature == 'waveform':
                # Plot waveform
                librosa.display.waveshow(y, sr=sr, ax=axes[i], alpha=0.7)
                axes[i].set_title("Waveform")
                
            elif feature == 'spectrogram':
                # Plot spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[i])
                fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
                axes[i].set_title("Spectrogram")
                
            elif feature == 'mel':
                # Plot mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[i])
                fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
                axes[i].set_title("Mel Spectrogram")
                
            elif feature == 'mfcc':
                # Plot MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr)
                img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[i])
                fig.colorbar(img, ax=axes[i])
                axes[i].set_title("MFCCs")
                
            elif feature == 'chroma':
                # Plot chromagram
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[i])
                fig.colorbar(img, ax=axes[i])
                axes[i].set_title("Chromagram")
                
            elif feature == 'pitch':
                # Plot pitch
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
                )
                times = librosa.times_like(f0, sr=sr)
                axes[i].plot(times, f0, color='#e74c3c', alpha=0.9, linewidth=2)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylabel("Frequency (Hz)")
                axes[i].set_title("Pitch (F0) Contour")
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_feature_comparison(self, audio_files: Dict[str, Tuple[np.ndarray, int]], 
                               feature: str = 'mel', 
                               title: str = "Feature Comparison", 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare a specific feature across multiple audio files.
        
        Args:
            audio_files: Dictionary mapping names to (audio_data, sample_rate) tuples
            feature: Feature to compare ('mel', 'mfcc', 'chroma', 'spectrogram')
            title: Main plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        num_files = len(audio_files)
        fig, axes = plt.subplots(num_files, 1, figsize=(12, 4 * num_files), dpi=self.dpi)
        
        if num_files == 1:
            axes = [axes]
        
        for i, (name, (y, sr)) in enumerate(audio_files.items()):
            if feature == 'mel':
                # Plot mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[i])
                fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
                
            elif feature == 'mfcc':
                # Plot MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr)
                img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[i])
                fig.colorbar(img, ax=axes[i])
                
            elif feature == 'chroma':
                # Plot chromagram
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[i])
                fig.colorbar(img, ax=axes[i])
                
            elif feature == 'spectrogram':
                # Plot spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[i])
                fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
                
            elif feature == 'waveform':
                # Plot waveform
                librosa.display.waveshow(y, sr=sr, ax=axes[i], alpha=0.7)
            
            # Add title for each subplot
            axes[i].set_title(f"{name} - {feature.capitalize()}")
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_onset_strength(self, y: np.ndarray, sr: int, 
                           title: str = "Onset Strength", 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the onset strength envelope of an audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        
        # Plot onset envelope
        ax.plot(times, onset_env, color='#9b59b6', alpha=0.8, linewidth=2)
        
        # Detect peaks (onsets)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        ax.plot(times[peaks], onset_env[peaks], 'ro', alpha=0.8)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strength")
        ax.set_title(title)
        
        # Save if requested
        if save_path:
            plt.tight_layout()
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
    
    # Load audio file
    y, sr = librosa.load(librosa.example('brahms'), duration=15)
    
    # Create visualizer
    viz = AudioVisualizer()
    
    # Plot waveform
    fig = viz.plot_waveform(y, sr, title="Brahms Waveform")
    
    # Plot multiple features
    fig = viz.plot_multi_feature(y, sr, features=['waveform', 'spectrogram', 'mel', 'mfcc'],
                                 title="Audio Features - Brahms")
    
    plt.show()

import os
import glob
import numpy as np
import pandas as pd
import json
import csv
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
from tqdm import tqdm


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("AudioInsight")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_id() -> str:
    """
    Generate a unique ID for files and models.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """
    Get current timestamp string for file naming.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def list_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    List all audio files in a directory with specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: ['.mp3', '.wav', '.flac', '.ogg'])
        
    Returns:
        List of paths to audio files
    """
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    
    audio_files = []
    
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
    
    return sorted(audio_files)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame as CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data.to_csv(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def process_audio_batch(audio_files: List[str], 
                       processor_func: callable, 
                       output_dir: Optional[str] = None, 
                       show_progress: bool = True) -> Dict[str, Any]:
    """
    Process a batch of audio files with a given function.
    
    Args:
        audio_files: List of audio file paths
        processor_func: Function to process each file (should take filepath and return result)
        output_dir: Directory to save results (optional)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping filenames to results
    """
    results = {}
    
    # Create iterator with progress bar if requested
    iterator = tqdm(audio_files) if show_progress else audio_files
    
    for filepath in iterator:
        try:
            filename = os.path.basename(filepath)
            result = processor_func(filepath)
            results[filename] = result
            
            # Save individual result if output_dir specified
            if output_dir:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save result (format depends on result type)
                if isinstance(result, pd.DataFrame):
                    result.to_csv(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv"), index=False)
                elif isinstance(result, dict):
                    with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json"), 'w') as f:
                        json.dump(result, f, indent=2)
                # Add more formats as needed
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return results


def segment_audio(y: np.ndarray, sr: int, 
                 segment_length: float = 5.0, 
                 hop_length: float = 2.5,
                 min_segment_length: float = 1.0) -> List[np.ndarray]:
    """
    Segment audio into overlapping chunks.
    
    Args:
        y: Audio time series
        sr: Sample rate
        segment_length: Length of each segment in seconds
        hop_length: Hop size between segments in seconds
        min_segment_length: Minimum segment length to include
        
    Returns:
        List of audio segments
    """
    # Convert seconds to samples
    segment_samples = int(segment_length * sr)
    hop_samples = int(hop_length * sr)
    min_segment_samples = int(min_segment_length * sr)
    
    # Calculate number of segments
    num_segments = 1 + (len(y) - segment_samples) // hop_samples
    
    segments = []
    
    # Extract segments
    for i in range(max(1, num_segments)):
        start = i * hop_samples
        end = start + segment_samples
        
        # Handle last segment
        if end > len(y):
            # Only include if it meets minimum length
            if len(y) - start >= min_segment_samples:
                segment = y[start:]
                # Pad if necessary
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))
                segments.append(segment)
        else:
            segments.append(y[start:end])
    
    return segments


def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, 
                          title: str = "Feature Importance", 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance for a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and get top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Plot horizontal bar chart
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
    
    return fig


# Example usage
if __name__ == "__main__":
    # Set up logging
    logger = setup_logging("logs/example.log")
    logger.info("Helper functions module loaded")
    
    # List audio files
    audio_dir = "data/raw"
    if os.path.exists(audio_dir):
        audio_files = list_audio_files(audio_dir)
        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

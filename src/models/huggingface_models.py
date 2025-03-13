import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoProcessor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
import librosa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


class HuggingFaceAudioModel:
    """
    Class for using HuggingFace's audio models for emotion and accent detection.
    Supports fine-tuning and inference with pretrained models.
    """
    
    # Model types and their default pretrained models
    MODEL_TYPES = {
        'emotion': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
        'accent': 'anton-l/wav2vec2-base-superb-sid',  # Speaker/accent identification
        'general': 'facebook/wav2vec2-base-960h'  # General speech recognition model for fine-tuning
    }
    
    # Default label maps for pretrained emotion model
    EMOTION_LABELS = {
        0: 'anger', 
        1: 'happiness', 
        2: 'sadness', 
        3: 'fear', 
        4: 'neutral'
    }
    
    def __init__(self, task: str = 'emotion', pretrained_model: Optional[str] = None, 
                 labels: Optional[Dict[int, str]] = None, device: Optional[str] = None):
        """
        Initialize the HuggingFace audio model.
        
        Args:
            task: Task type ('emotion', 'accent', or 'general')
            pretrained_model: HuggingFace model name or path (if None, uses default for task)
            labels: Label mapping (if None, uses default for task)
            device: Torch device ('cuda', 'cpu', or None for auto-detection)
        """
        self.task = task
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set up model name
        if pretrained_model is None:
            if task not in self.MODEL_TYPES:
                raise ValueError(f"Unknown task: {task}. Choose from {list(self.MODEL_TYPES.keys())}")
            self.model_name = self.MODEL_TYPES[task]
        else:
            self.model_name = pretrained_model
        
        # Set up labels
        if labels is None:
            if task == 'emotion':
                self.labels = self.EMOTION_LABELS
            else:
                self.labels = {}  # Will be set during fine-tuning
        else:
            self.labels = labels
        
        # Initialize processor and model
        self.processor = None
        self.model = None
        self.sampling_rate = 16000  # Default for Wav2Vec2
        
        # Load model and processor if not fine-tuning from scratch
        if 'facebook/wav2vec2' not in self.model_name:
            self._load_model_and_processor()
    
    def _load_model_and_processor(self) -> None:
        """
        Load model and processor from HuggingFace.
        """
        try:
            # Try loading with AutoProcessor first
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except:
            # Fall back to Wav2Vec2Processor
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            except:
                # Fall back to just the feature extractor
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        
        # Load model
        if self.task in ['emotion', 'accent']:
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        else:
            # For general speech, we'll fine-tune from base model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Get sampling rate from processor
        if hasattr(self.processor, 'feature_extractor'):
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            self.sampling_rate = getattr(self.processor, 'sampling_rate', 16000)
        
        print(f"Model '{self.model_name}' loaded successfully on {self.device}")
    
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> torch.Tensor:
        """
        Preprocess audio data for the model.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            Preprocessed inputs ready for the model
        """
        # Resample if needed
        if sr != self.sampling_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=sr, target_sr=self.sampling_rate
            )
        
        # Process audio with the processor
        inputs = self.processor(
            audio_data, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        return inputs
    
    def prepare_dataset(self, 
                        audio_paths: List[str], 
                        labels: Optional[List[Union[str, int]]] = None,
                        label_to_id: Optional[Dict[str, int]] = None) -> Dataset:
        """
        Prepare a HuggingFace dataset from audio files.
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels (optional, for training)
            label_to_id: Mapping from label string to ID (optional)
            
        Returns:
            HuggingFace Dataset
        """
        # Create mapping from label names to IDs if not provided
        if labels is not None and label_to_id is None:
            unique_labels = sorted(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            id_to_label = {i: label for label, i in label_to_id.items()}
            self.labels = id_to_label
        
        # Create dataset dictionary
        dataset_dict = {"audio": audio_paths}
        
        if labels is not None:
            # Convert string labels to IDs if needed
            if isinstance(labels[0], str) and label_to_id is not None:
                label_ids = [label_to_id[label] for label in labels]
            else:
                label_ids = labels
            
            dataset_dict["label"] = label_ids
        
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Add audio loading
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        
        return dataset
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess a dataset for training or inference.
        
        Args:
            dataset: HuggingFace Dataset
            
        Returns:
            Preprocessed dataset
        """
        # Define preprocessing function
        def preprocess_function(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            inputs = self.processor(
                audio_arrays, 
                sampling_rate=self.sampling_rate, 
                padding=True,
                return_tensors="pt"
            )
            return inputs
        
        # Apply preprocessing
        dataset = dataset.map(preprocess_function, batched=True)
        
        return dataset
    
    def fine_tune(self, 
                 dataset: Dataset,
                 num_labels: int,
                 epochs: int = 10,
                 batch_size: int = 8,
                 learning_rate: float = 5e-5,
                 output_dir: str = "results",
                 evaluation_strategy: str = "epoch") -> Dict[str, float]:
        """
        Fine-tune the model on a dataset.
        
        Args:
            dataset: HuggingFace Dataset
            num_labels: Number of labels for classification
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_dir: Directory to save results
            evaluation_strategy: When to evaluate ('epoch', 'steps', or 'no')
            
        Returns:
            Dictionary with training metrics
        """
        # Initialize model for fine-tuning if not already loaded
        if self.model is None:
            if self.task in ['emotion', 'accent']:
                self.model = AutoModelForAudioClassification.from_pretrained(
                    self.model_name, 
                    num_labels=num_labels
                )
            else:
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels
                )
            
            # Move model to device
            self.model = self.model.to(self.device)
        
        # Initialize processor if not already loaded
        if self.processor is None:
            self._load_model_and_processor()
        
        # Split dataset into train and validation
        dataset = dataset.train_test_split(test_size=0.2)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy=evaluation_strategy,
            save_strategy=evaluation_strategy,
            load_best_model_at_end=True,
            push_to_hub=False
        )
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        
        return metrics
    
    def predict(self, audio_data: np.ndarray, sr: int) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict from raw audio data.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before prediction.")
        
        # Preprocess audio
        inputs = self.preprocess_audio(audio_data, sr)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        
        # Get predicted label
        predicted_label_id = predictions[0]
        if predicted_label_id in self.labels:
            predicted_label = self.labels[predicted_label_id]
        else:
            predicted_label = str(predicted_label_id)
        
        # Create probability dictionary
        probs_dict = {}
        for i, prob in enumerate(probabilities[0]):
            if i in self.labels:
                probs_dict[self.labels[i]] = float(prob)
            else:
                probs_dict[str(i)] = float(prob)
        
        return {
            'label': predicted_label,
            'probabilities': probs_dict
        }
    
    def batch_predict(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict for a batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction results
        """
        # Create dataset
        dataset = self.prepare_dataset(audio_paths)
        
        # Preprocess dataset
        dataset = self.preprocess_dataset(dataset)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Process outputs
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                
                # Add predictions
                for i, label_id in enumerate(predicted_labels):
                    if label_id in self.labels:
                        label = self.labels[label_id]
                    else:
                        label = str(label_id)
                    
                    probs = {}
                    for j, prob in enumerate(probabilities[i]):
                        if j in self.labels:
                            probs[self.labels[j]] = float(prob)
                        else:
                            probs[str(j)] = float(prob)
                    
                    predictions.append({
                        'label': label,
                        'probabilities': probs
                    })
        
        return predictions
    
    def save_model(self, output_dir: str) -> None:
        """
        Save model and processor.
        
        Args:
            output_dir: Directory to save to
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before saving.")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and processor
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # Save label mapping
        if self.labels:
            import json
            with open(os.path.join(output_dir, "label_map.json"), "w") as f:
                json.dump(self.labels, f)
        
        print(f"Model and processor saved to {output_dir}")
    
    def load_saved_model(self, model_dir: str) -> None:
        """
        Load a saved model and processor.
        
        Args:
            model_dir: Directory containing saved model and processor
        """
        # Load model
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_dir)
        except:
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
            except:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        
        # Load label mapping if available
        label_map_path = os.path.join(model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path, "r") as f:
                self.labels = json.load(f)
                # Convert string keys to integers
                self.labels = {int(k): v for k, v in self.labels.items()}
        
        # Get sampling rate from processor
        if hasattr(self.processor, 'feature_extractor'):
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            self.sampling_rate = getattr(self.processor, 'sampling_rate', 16000)
        
        print(f"Model and processor loaded from {model_dir}")


# Example usage
if __name__ == "__main__":
    # Example: Load pretrained emotion model
    emotion_model = HuggingFaceAudioModel(task='emotion')
    
    # Example: Load audio
    audio_path = "path/to/audio.wav"
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Predict emotion
    result = emotion_model.predict(y, sr)
    print(f"Predicted emotion: {result['label']}")
    print(f"Probabilities: {result['probabilities']}")

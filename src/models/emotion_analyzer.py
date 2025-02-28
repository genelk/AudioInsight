import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.feature_extraction import FeatureExtractor


class EmotionAnalyzer:
    """
    Model for emotion detection in audio.
    Supports both traditional ML models and integration with HuggingFace models.
    """
    
    # Define emotion categories
    EMOTIONS = {
        'primary': ['anger', 'happiness', 'sadness', 'fear', 'neutral'],
        'secondary': ['distress', 'anxiety', 'excitement', 'confidence', 'neutral']
    }
    
    def __init__(self, model_type: str = 'rf', use_primary_emotions: bool = True, 
                 model_path: Optional[str] = None):
        """
        Initialize the emotion analyzer.
        
        Args:
            model_type: Type of model to use ('rf', 'svm', 'mlp', 'huggingface')
            use_primary_emotions: Whether to use primary or secondary emotions
            model_path: Path to a pre-trained model (optional)
        """
        self.model_type = model_type
        self.use_primary_emotions = use_primary_emotions
        self.emotions = self.EMOTIONS['primary'] if use_primary_emotions else self.EMOTIONS['secondary']
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.feature_columns = None
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_model(self) -> Union[Pipeline, None]:
        """
        Create a new model based on model_type.
        
        Returns:
            Scikit-learn Pipeline or None
        """
        if self.model_type == 'rf':
            # Random Forest model
            classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=None, 
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            # Support Vector Machine
            classifier = SVC(
                probability=True, 
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'mlp':
            # Neural Network (MLP)
            classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                activation='relu',
                solver='adam', 
                alpha=0.0001, 
                batch_size='auto',
                learning_rate='adaptive', 
                max_iter=500, 
                random_state=42
            )
        elif self.model_type == 'huggingface':
            # HuggingFace model is initialized separately
            return None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline with standard scaler
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
    def prepare_data(self, features: pd.DataFrame, labels: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for model training or prediction.
        
        Args:
            features: DataFrame of extracted features
            labels: Series of emotion labels (optional, for training)
            
        Returns:
            Tuple of (X, y) where y may be None for prediction
        """
        # Store feature columns during training
        if labels is not None and self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        
        # Ensure features match expected columns during prediction
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(features.columns)
            if missing_cols:
                for col in missing_cols:
                    features[col] = 0  # Fill missing with zeros
            
            # Keep only required columns in the right order
            X = features[self.feature_columns].values
        else:
            # First time prediction without training, use all features
            X = features.values
        
        # Prepare labels if provided
        y = None
        if labels is not None:
            y = labels.values
        
        return X, y
    
    def extract_emotion_features(self, audio_data: np.ndarray, sr: int) -> pd.DataFrame:
        """
        Extract features relevant for emotion detection.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            DataFrame with extracted features
        """
        # Extract features using FeatureExtractor
        features = self.feature_extractor.extract_features(
            audio_data, sr, feature_types=['mfcc', 'spectral', 'prosodic', 'emotion']
        )
        
        # Convert to DataFrame
        return self.feature_extractor.features_to_dataframe(features)
    
    def train(self, features: pd.DataFrame, labels: pd.Series, 
              test_size: float = 0.2, optimize_hyperparams: bool = False) -> Dict[str, float]:
        """
        Train the emotion detection model.
        
        Args:
            features: DataFrame of extracted features
            labels: Series of emotion labels
            test_size: Proportion of data to use for testing
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize model if not already done
        if self.model is None and self.model_type != 'huggingface':
            self.model = self._create_model()
        
        # Prepare data
        X, y = self.prepare_data(features, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train models differently based on type
        if self.model_type == 'huggingface':
            # For HuggingFace models, we'll implement fine-tuning in huggingface_models.py
            # Here we'll just store features for later use
            self.feature_columns = features.columns.tolist()
            return {"message": "HuggingFace model training delegated to huggingface_models.py"}
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            if self.model_type == 'rf':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
                    'classifier__kernel': ['rbf', 'linear']
                }
            elif self.model_type == 'mlp':
                param_grid = {
                    'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'classifier__alpha': [0.0001, 0.001, 0.01],
                    'classifier__learning_rate': ['constant', 'adaptive']
                }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print classification report
        report = classification_report(y_test, y_pred, target_names=self.emotions, output_dict=True)
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        return metrics
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict emotion from features.
        
        Args:
            features: DataFrame of extracted features
            
        Returns:
            Dictionary with predicted emotion and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Prepare features
        X, _ = self.prepare_data(features)
        
        # Make prediction
        emotion_idx = self.model.predict(X)[0]
        emotion = self.emotions[emotion_idx] if isinstance(emotion_idx, (np.integer, int)) else emotion_idx
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[0]
        emotion_probs = {self.emotions[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return {
            'emotion': emotion,
            'probabilities': emotion_probs
        }
    
    def predict_from_audio(self, audio_data: np.ndarray, sr: int) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict emotion directly from audio data.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with predicted emotion and probabilities
        """
        # Extract features
        features = self.extract_emotion_features(audio_data, sr)
        
        # Predict
        return self.predict(features)
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and feature columns
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'emotions': self.emotions,
            'model_type': self.model_type,
            'use_primary_emotions': self.use_primary_emotions
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        # Load model and metadata
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.emotions = model_data['emotions']
        self.model_type = model_data['model_type']
        self.use_primary_emotions = model_data['use_primary_emotions']
        
        print(f"Model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    import librosa
    from src.data.audio_loader import AudioLoader
    
    # Load audio file
    loader = AudioLoader()
    audio, sr = loader.load_file("path/to/audio.wav")
    
    # Create and train model
    emotion_analyzer = EmotionAnalyzer(model_type='rf')
    
    # Extract features
    features = emotion_analyzer.extract_emotion_features(audio, sr)
    
    # If we had labels, we could train
    # emotion_analyzer.train(features, labels)
    
    # For demonstration, we'll load a pre-trained model
    # emotion_analyzer.load_model("models/emotion_model.joblib")
    
    # Predict
    # result = emotion_analyzer.predict_from_audio(audio, sr)
    # print(f"Predicted emotion: {result['emotion']}")
    # print(f"Probabilities: {result['probabilities']}")

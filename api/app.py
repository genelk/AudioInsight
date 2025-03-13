import os
import uuid
import json
import traceback
from typing import Dict, List, Union, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import pandas as pd
import librosa
from pydantic import BaseModel

# Import local modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.audio_loader import AudioLoader
from src.data.feature_extraction import FeatureExtractor
from src.models.emotion_analyzer import EmotionAnalyzer
from src.models.accent_detector import AccentDetector
from src.models.huggingface_models import HuggingFaceAudioModel
from src.utils.helpers import ensure_dir, save_json, get_timestamp, generate_id

# Create directories for uploads and results
UPLOAD_DIR = Path('data/uploads')
RESULTS_DIR = Path('data/results')
MODEL_DIR = Path('data/models')

ensure_dir(UPLOAD_DIR)
ensure_dir(RESULTS_DIR)
ensure_dir(MODEL_DIR)

# Initialize FastAPI app
app = FastAPI(
    title="AudioInsight API",
    description="Audio analysis for emotion and accent detection",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
audio_loader = AudioLoader()
feature_extractor = FeatureExtractor()

# Models will be loaded on-demand
emotion_models = {}
accent_models = {}


# Define request/response models
class AnalysisRequest(BaseModel):
    analysis_type: str  # 'emotion', 'accent', or 'both'
    model_type: str = 'rf'  # 'rf', 'svm', 'mlp', or 'huggingface'
    detailed: bool = False


class AnalysisResponse(BaseModel):
    file_id: str
    filename: str
    duration: float
    analysis_type: str
    results: Dict
    features: Optional[Dict] = None


# Helper function to load models
def get_emotion_model(model_type: str = 'rf'):
    """Load or retrieve emotion model by type."""
    if model_type not in emotion_models:
        # Check if saved model exists
        model_path = MODEL_DIR / f"emotion_{model_type}.joblib"
        
        if model_path.exists():
            # Load saved model
            model = EmotionAnalyzer(model_type=model_type, model_path=str(model_path))
        else:
            # Initialize new model
            model = EmotionAnalyzer(model_type=model_type)
            # Model will need to be trained before use
        
        emotion_models[model_type] = model
    
    return emotion_models[model_type]


def get_accent_model(model_type: str = 'rf'):
    """Load or retrieve accent model by type."""
    if model_type not in accent_models:
        # Check if saved model exists
        model_path = MODEL_DIR / f"accent_{model_type}.joblib"
        
        if model_path.exists():
            # Load saved model
            model = AccentDetector(model_type=model_type, model_path=str(model_path))
        else:
            # Initialize new model
            model = AccentDetector(model_type=model_type)
            # Model will need to be trained before use
        
        accent_models[model_type] = model
    
    return accent_models[model_type]


def get_huggingface_model(task: str = 'emotion'):
    """Load or retrieve HuggingFace model by task."""
    model_key = f"huggingface_{task}"
    
    if model_key not in emotion_models and model_key not in accent_models:
        # Initialize new model
        model = HuggingFaceAudioModel(task=task)
        
        if task == 'emotion':
            emotion_models[model_key] = model
        else:
            accent_models[model_key] = model
    
    if task == 'emotion':
        return emotion_models[model_key]
    else:
        return accent_models[model_key]


# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to AudioInsight API. Use /docs for API documentation."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file."""
    file_id = generate_id()
    filename = file.filename
    
    # Create file path
    file_path = UPLOAD_DIR / f"{file_id}_{filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return {
            "file_id": file_id,
            "filename": filename,
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/analyze/{file_id}")
async def analyze_audio(
    file_id: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze an audio file for emotion or accent."""
    # Find the uploaded file
    file_path = None
    for f in UPLOAD_DIR.glob(f"{file_id}_*"):
        file_path = f
        break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Load audio file
        y, sr = audio_loader.load_file(str(file_path))
        
        # Extract features
        if request.analysis_type == 'emotion':
            features = feature_extractor.extract_features(
                y, sr, feature_types=['mfcc', 'spectral', 'prosodic', 'emotion']
            )
        elif request.analysis_type == 'accent':
            features = feature_extractor.extract_features(
                y, sr, feature_types=['mfcc', 'spectral', 'rhythm', 'accent']
            )
        else:  # both
            features = feature_extractor.extract_features(
                y, sr, feature_types=['mfcc', 'spectral', 'rhythm', 'prosodic', 'emotion', 'accent']
            )
        
        # Convert features to DataFrame
        features_df = feature_extractor.features_to_dataframe(features)
        
        # Initialize results dictionary
        results = {
            "status": "processing",
            "file_id": file_id,
            "filename": file_path.name.replace(f"{file_id}_", ""),
            "duration": len(y) / sr,
            "analysis_type": request.analysis_type
        }
        
        # Create result file
        result_id = generate_id()
        result_path = RESULTS_DIR / f"{result_id}.json"
        save_json(results, str(result_path))
        
        # Schedule background processing
        background_tasks.add_task(
            process_analysis,
            y, sr, features, features_df, request.analysis_type, 
            request.model_type, str(result_path), request.detailed
        )
        
        return {
            "result_id": result_id,
            "status": "processing",
            "message": "Analysis started. Check status with /status/{result_id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")


async def process_analysis(
    y: np.ndarray,
    sr: int,
    features: Dict,
    features_df: pd.DataFrame,
    analysis_type: str,
    model_type: str,
    result_path: str,
    detailed: bool
):
    """Process analysis in background."""
    try:
        results = {}
        
        # Perform analysis based on type
        if analysis_type == 'emotion' or analysis_type == 'both':
            if model_type == 'huggingface':
                model = get_huggingface_model('emotion')
                emotion_result = model.predict(y, sr)
            else:
                model = get_emotion_model(model_type)
                emotion_result = model.predict(features_df)
            
            results['emotion'] = emotion_result
        
        if analysis_type == 'accent' or analysis_type == 'both':
            if model_type == 'huggingface':
                model = get_huggingface_model('accent')
                accent_result = model.predict(y, sr)
            else:
                model = get_accent_model(model_type)
                accent_result = model.predict(features_df)
            
            results['accent'] = accent_result
        
        # Load current result file
        with open(result_path, 'r') as f:
            current_results = json.load(f)
        
        # Update results
        current_results['status'] = 'completed'
        current_results['results'] = results
        
        # Add detailed features if requested
        if detailed:
            # Convert NumPy arrays to lists for JSON serialization
            serializable_features = {}
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    if value.size < 1000:  # Only include small arrays
                        serializable_features[key] = value.tolist()
                else:
                    serializable_features[key] = value
            
            current_results['features'] = serializable_features
        
        # Save updated results
        save_json(current_results, result_path)
    
    except Exception as e:
        # Update result file with error
        with open(result_path, 'r') as f:
            current_results = json.load(f)
        
        current_results['status'] = 'error'
        current_results['error'] = str(e)
        current_results['traceback'] = traceback.format_exc()
        
        save_json(current_results, result_path)


@app.get("/status/{result_id}")
async def check_status(result_id: str):
    """Check the status of an analysis."""
    result_path = RESULTS_DIR / f"{result_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving result: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    models = []
    
    # List pre-trained models
    for model_file in MODEL_DIR.glob("*.joblib"):
        model_info = {
            "id": model_file.stem,
            "type": "pretrained",
            "path": str(model_file)
        }
        models.append(model_info)
    
    # Add HuggingFace models
    models.append({
        "id": "huggingface_emotion",
        "type": "huggingface",
        "task": "emotion"
    })
    
    models.append({
        "id": "huggingface_accent",
        "type": "huggingface",
        "task": "accent"
    })
    
    return {"models": models}


# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

#!/usr/bin/env python3
"""
FastAPI Application
Provides REST API endpoints for sentiment analysis of mental health text.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import yaml
import sys
import json

# Add parent directory to path for importing local modules
sys.path.append(str(Path(__file__).parent.parent))
from models.sentiment_analyzer import SentimentAnalyzer
from preprocessing.text_processor import TextProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "config/config.yaml") -> dict:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)['api']
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Text Sentiment Analysis API",
    description="API for analyzing sentiment in mental health-related text",
    version="1.0.0"
)

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
text_processor = TextProcessor()

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="The text to analyze")
    
class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")

class SentimentResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    scores: Dict[str, Any]
    entities: List[tuple]
    
class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    summary: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Mental Health Text Sentiment Analysis API",
        "version": "1.0.0",
        "description": "Analyze sentiment in mental health-related text"
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(input_data: TextInput):
    """Analyze sentiment of a single text."""
    try:
        # Preprocess text
        processed = text_processor.process_text(input_data.text)
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze_text(processed['cleaned_text'])
        
        return SentimentResponse(
            text=input_data.text,
            cleaned_text=processed['cleaned_text'],
            sentiment=sentiment_result['combined_sentiment'],
            scores=sentiment_result,
            entities=processed['entities']
        )
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts."""
    try:
        results = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for text in input_data.texts:
            # Preprocess text
            processed = text_processor.process_text(text)
            
            # Analyze sentiment
            sentiment_result = sentiment_analyzer.analyze_text(processed['cleaned_text'])
            
            # Update counts
            sentiment_counts[sentiment_result['combined_sentiment']] += 1
            
            results.append(SentimentResponse(
                text=text,
                cleaned_text=processed['cleaned_text'],
                sentiment=sentiment_result['combined_sentiment'],
                scores=sentiment_result,
                entities=processed['entities']
            ))
        
        # Calculate summary statistics
        total = len(input_data.texts)
        summary = {
            'total_texts': total,
            'sentiment_distribution': {
                sentiment: {
                    'count': count,
                    'percentage': (count / total) * 100 if total > 0 else 0
                }
                for sentiment, count in sentiment_counts.items()
            }
        }
        
        return BatchSentimentResponse(results=results, summary=summary)
    except Exception as e:
        logger.error(f"Error analyzing batch texts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def start_server():
    """Start the FastAPI server."""
    config = load_config()
    
    uvicorn.run(
        "app:app",
        host=config['host'],
        port=config['port'],
        workers=config['workers'],
        reload=True
    )

if __name__ == "__main__":
    start_server() 
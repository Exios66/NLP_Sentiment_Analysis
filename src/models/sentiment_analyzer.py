#!/usr/bin/env python3
"""
Sentiment Analysis Module
Implements multiple sentiment analysis approaches including VADER and BERT
for mental health text analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """PyTorch dataset for text data."""
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])

        return item

class SentimentAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the sentiment analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize VADER
        if self.config['models']['vader']['enabled']:
            self.vader = SentimentIntensityAnalyzer()
        
        # Initialize BERT
        self.bert_config = self.config['models']['bert']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.bert_config['model_name'],
            num_labels=3  # positive, negative, neutral
        ).to(self.device)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['sentiment_analysis']
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }

    def analyze_bert(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """Analyze sentiment using BERT."""
        self.model.eval()
        dataset = TextDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing with BERT"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                for probs in probabilities:
                    results.append({
                        'negative': probs[0].item(),
                        'neutral': probs[1].item(),
                        'positive': probs[2].item()
                    })
        
        return results

    def fine_tune(self, texts: List[str], labels: List[int],
                 val_size: float = 0.2, save_path: Optional[Path] = None):
        """Fine-tune BERT model on custom data."""
        # Prepare datasets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, random_state=42
        )
        
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.bert_config['batch_size'],
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.bert_config['batch_size']
        )
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.bert_config['learning_rate'])
        )
        
        total_steps = len(train_dataloader) * self.bert_config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_accuracy = 0
        for epoch in range(self.bert_config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.bert_config['epochs']}")
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_dataloader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_accuracy = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_accuracy += (predictions == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            accuracy = val_accuracy / len(val_dataset)
            
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Accuracy: {accuracy:.4f}")
            
            # Save best model
            if save_path and accuracy > best_accuracy:
                best_accuracy = accuracy
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")

    def analyze_text(self, text: str) -> Dict[str, Union[Dict[str, float], str]]:
        """Analyze sentiment using both VADER and BERT."""
        # VADER analysis
        vader_scores = self.analyze_vader(text)
        
        # BERT analysis
        bert_scores = self.analyze_bert([text])[0]
        
        # Combine results
        result = {
            'vader': vader_scores,
            'bert': bert_scores,
            'combined_sentiment': self._get_combined_sentiment(vader_scores, bert_scores)
        }
        
        return result

    def _get_combined_sentiment(self, vader_scores: Dict[str, float],
                              bert_scores: Dict[str, float]) -> str:
        """Combine VADER and BERT scores to get final sentiment."""
        # Weight the scores (can be adjusted based on performance)
        vader_weight = 0.4
        bert_weight = 0.6
        
        # Calculate weighted compound score
        vader_compound = vader_scores['compound']
        bert_compound = bert_scores['positive'] - bert_scores['negative']
        
        combined_score = (vader_compound * vader_weight) + (bert_compound * bert_weight)
        
        # Determine sentiment based on combined score
        if combined_score >= 0.05:
            return 'positive'
        elif combined_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def process_file(self, input_path: Path, output_path: Optional[Path] = None,
                    text_column: str = 'cleaned_text') -> pd.DataFrame:
        """Process texts from a file and save results."""
        try:
            # Read input file
            df = pd.read_csv(input_path)
            
            # Analyze sentiments
            results = []
            for text in tqdm(df[text_column], desc="Analyzing sentiments"):
                result = self.analyze_text(text)
                results.append(result)
            
            # Add results to DataFrame
            df['sentiment_scores'] = results
            df['sentiment'] = [r['combined_sentiment'] for r in results]
            
            # Save processed data if output path is provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Sentiment analysis results saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

def main():
    """Main function to run the sentiment analyzer."""
    try:
        analyzer = SentimentAnalyzer()
        
        # Process the latest preprocessed data
        processed_dir = Path("data/processed")
        analyzed_dir = Path("data/processed/sentiment")
        analyzed_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the latest preprocessed file
        latest_file = max(processed_dir.glob("processed_*.csv"), key=lambda x: x.stat().st_mtime)
        
        # Process the file
        output_path = analyzed_dir / f"sentiment_{latest_file.name}"
        analyzer.process_file(latest_file, output_path)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main() 
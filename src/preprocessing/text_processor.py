#!/usr/bin/env python3
"""
Text Preprocessing Module
Handles all text preprocessing tasks including cleaning, tokenization,
and feature extraction for mental health text analysis.
"""

import re
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path

import nltk

import spacy
import pandas as pd
import yaml
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the text processor with configuration."""
        self.config = self._load_config(config_path)
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile("["
            u"U0001F600-U0001F64F"  # emoticons
            u"U0001F300-U0001F5FF"  # symbols & pictographs
            u"U0001F680-U0001F6FF"  # transport & map symbols
            u"U0001F1E0-U0001F1FF"  # flags (iOS)
            u"U00002702-U000027B0"
            u"U000024C2-U0001F251"
            "]+", flags=re.UNICODE)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['preprocessing']
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing unwanted elements."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase if configured
        if self.config['text_cleaning']['lowercase']:
            text = text.lower()

        # Remove URLs if configured
        if self.config['text_cleaning']['remove_urls']:
            text = self.url_pattern.sub('', text)

        # Remove emojis if configured
        if self.config['text_cleaning']['remove_emojis']:
            text = self.emoji_pattern.sub('', text)

        # Remove special characters if configured
        if self.config['text_cleaning']['remove_special_chars']:
            text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        tokens = word_tokenize(text)
        
        # Filter tokens based on configuration
        if self.config['tokenization']['remove_stopwords']:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        # Filter by minimum token length
        tokens = [t for t in tokens if len(t) >= self.config['tokenization']['min_token_length']]
        
        # Lemmatize if configured
        if self.config['tokenization']['lemmatize']:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens

    def process_text(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """Process a single text through the entire pipeline."""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        # Get spaCy analysis
        doc = self.nlp(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'pos_tags': [(token.text, token.pos_) for token in doc]
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process all texts in a DataFrame."""
        logger.info(f"Processing {len(df)} texts...")
        
        processed_texts = []
        for text in tqdm(df[text_column], desc="Processing texts"):
            processed = self.process_text(text)
            processed_texts.append(processed)
        
        # Create new columns for processed data
        df['cleaned_text'] = [p['cleaned_text'] for p in processed_texts]
        df['tokens'] = [p['tokens'] for p in processed_texts]
        df['entities'] = [p['entities'] for p in processed_texts]
        df['noun_phrases'] = [p['noun_phrases'] for p in processed_texts]
        df['pos_tags'] = [p['pos_tags'] for p in processed_texts]
        
        return df

    def process_file(self, input_path: Path, output_path: Optional[Path] = None,
                    text_column: str = 'text') -> pd.DataFrame:
        """Process texts from a file and save results."""
        try:
            # Read input file
            df = pd.read_csv(input_path)
            
            # Process the texts
            processed_df = self.process_dataframe(df, text_column)
            
            # Save processed data if output path is provided
            if output_path:
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to {output_path}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

def main():
    """Main function to run the text processor."""
    try:
        processor = TextProcessor()
        
        # Process the latest collected Reddit data
        data_dir = Path("data/raw/reddit")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the latest Reddit data file
        latest_file = max(data_dir.glob("reddit_data_*.csv"), key=lambda x: x.stat().st_mtime)
        
        # Process the file
        output_path = processed_dir / f"processed_{latest_file.name}"
        processor.process_file(latest_file, output_path)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main() 
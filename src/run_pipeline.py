#!/usr/bin/env python3
"""
Pipeline Runner
Executes the complete sentiment analysis pipeline from data collection to visualization.
"""

import logging
import argparse
from pathlib import Path
import time
from typing import Optional, Dict, Any
import pandas as pd

from data_collection.reddit_scraper import RedditCollector
from data_collection.custom_data_loader import CustomDataLoader
from preprocessing.text_processor import TextProcessor
from models.sentiment_analyzer import SentimentAnalyzer
from visualization.visualizer import SentimentVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        """Initialize pipeline components."""
        self.data_collector = RedditCollector()
        self.custom_loader = CustomDataLoader()
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.visualizer = SentimentVisualizer()

    def run(self, input_mode: str = 'reddit', input_file: Optional[Path] = None,
            text_column: str = 'text', label_column: Optional[str] = None,
            skip_collection: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            input_mode: Either 'reddit' or 'custom'
            input_file: Path to input file for custom mode
            text_column: Name of text column for custom input
            label_column: Name of label column for custom input
            skip_collection: Whether to skip data collection (for reddit mode)
        
        Returns:
            Dict containing paths to output files
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Data Collection/Loading
            if input_mode == 'reddit':
                if not skip_collection:
                    logger.info("Step 1: Collecting data from Reddit...")
                    self.data_collector.run_collection()
                    data_dir = Path("data/raw/reddit")
                    latest_raw_file = max(data_dir.glob("reddit_data_*.csv"),
                                        key=lambda x: x.stat().st_mtime)
                else:
                    logger.info("Skipping Reddit data collection...")
                    if input_file:
                        latest_raw_file = input_file
                    else:
                        data_dir = Path("data/raw/reddit")
                        latest_raw_file = max(data_dir.glob("reddit_data_*.csv"),
                                            key=lambda x: x.stat().st_mtime)
            else:  # custom mode
                logger.info("Step 1: Loading custom data...")
                if not input_file:
                    raise ValueError("Input file is required for custom mode")
                latest_raw_file = self.custom_loader.load_and_process(
                    input_file,
                    text_column=text_column,
                    label_column=label_column
                )

            # Step 2: Text Preprocessing
            logger.info("Step 2: Preprocessing texts...")
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_file = processed_dir / f"processed_{timestamp}.csv"
            
            self.text_processor.process_file(latest_raw_file, processed_file)

            # Step 3: Sentiment Analysis
            logger.info("Step 3: Analyzing sentiments...")
            analyzed_dir = Path("data/processed/sentiment")
            analyzed_dir.mkdir(parents=True, exist_ok=True)
            analyzed_file = analyzed_dir / f"sentiment_{timestamp}.csv"
            
            df = self.sentiment_analyzer.process_file(processed_file, analyzed_file)

            # Step 4: Visualization
            logger.info("Step 4: Creating visualizations...")
            self.visualizer.create_all_visualizations(analyzed_file)

            # Step 5: Evaluate against true labels if provided
            if input_mode == 'custom' and label_column:
                logger.info("Step 5: Evaluating against true labels...")
                evaluation_results = self._evaluate_predictions(df)
                
                # Save evaluation results
                eval_dir = Path("data/evaluation")
                eval_dir.mkdir(parents=True, exist_ok=True)
                eval_file = eval_dir / f"evaluation_{timestamp}.json"
                
                import json
                with open(eval_file, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
            else:
                eval_file = None

            logger.info("Pipeline completed successfully!")
            
            return {
                'raw_data': latest_raw_file,
                'processed_data': processed_file,
                'sentiment_analysis': analyzed_file,
                'visualizations': Path('data/visualizations'),
                'evaluation': eval_file
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    def _evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate predictions against true labels.
        
        Args:
            df: DataFrame containing predictions and true labels
            
        Returns:
            Dict containing evaluation metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        true_labels = df['true_sentiment']
        predicted_labels = df['sentiment']
        
        # Calculate metrics
        classification_metrics = classification_report(
            true_labels,
            predicted_labels,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=['positive', 'neutral', 'negative']
        ).tolist()
        
        # Calculate accuracy for each sentiment
        sentiment_accuracy = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            mask = true_labels == sentiment
            if mask.any():
                accuracy = (predicted_labels[mask] == sentiment).mean()
                sentiment_accuracy[sentiment] = float(accuracy)
        
        return {
            'classification_report': classification_metrics,
            'confusion_matrix': conf_matrix,
            'sentiment_accuracy': sentiment_accuracy
        }

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the sentiment analysis pipeline")
    parser.add_argument(
        "--input-mode",
        choices=['reddit', 'custom'],
        default='reddit',
        help="Input data mode: 'reddit' or 'custom'"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file (required for custom mode, optional for reddit mode)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column for custom input"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        help="Name of label column for custom input"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip the data collection step (for reddit mode)"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if args.input_mode == 'custom' and not args.input_file:
        parser.error("--input-file is required when using custom mode")

    input_file = Path(args.input_file) if args.input_file else None

    pipeline = Pipeline()
    output_paths = pipeline.run(
        input_mode=args.input_mode,
        input_file=input_file,
        text_column=args.text_column,
        label_column=args.label_column,
        skip_collection=args.skip_collection
    )
    
    # Print output paths
    print("\nPipeline outputs:")
    for key, path in output_paths.items():
        if path:
            print(f"- {key}: {path}")

if __name__ == "__main__":
    main() 
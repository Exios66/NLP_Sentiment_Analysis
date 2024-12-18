#!/usr/bin/env python3
"""
Custom Data Loader
Handles loading and preprocessing of custom CSV files containing text responses.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomDataLoader:
    def __init__(self):
        """Initialize the custom data loader."""
        self.required_columns = ['text']
        self.output_dir = Path("data/raw/custom")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            bool: True if valid, raises ValueError if not
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True

    def standardize_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                            label_column: Optional[str] = None) -> pd.DataFrame:
        """
        Standardize the DataFrame to match the pipeline's expected format.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text data
            label_column: Optional name of the column containing sentiment labels
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Create a copy to avoid modifying the original
        standardized_df = df.copy()

        # Rename text column if different
        if text_column != 'text':
            standardized_df = standardized_df.rename(columns={text_column: 'text'})

        # Add timestamp if not present
        if 'created_utc' not in standardized_df.columns:
            standardized_df['created_utc'] = datetime.now().isoformat()

        # Add source identifier
        standardized_df['source'] = 'custom_input'

        # Handle sentiment labels if provided
        if label_column and label_column in df.columns:
            standardized_df = standardized_df.rename(columns={label_column: 'true_sentiment'})

        return standardized_df

    def load_and_process(self, input_path: Path, text_column: str = 'text',
                        label_column: Optional[str] = None) -> Path:
        """
        Load and process a custom CSV file.
        
        Args:
            input_path: Path to input CSV file
            text_column: Name of the column containing text data
            label_column: Optional name of the column containing sentiment labels
            
        Returns:
            Path: Path to the processed output file
        """
        try:
            logger.info(f"Loading custom data from {input_path}")
            
            # Read CSV file
            df = pd.read_csv(input_path)
            
            # Validate DataFrame
            self.validate_dataframe(df)
            
            # Standardize DataFrame
            processed_df = self.standardize_dataframe(df, text_column, label_column)
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"custom_data_{timestamp}.csv"
            processed_df.to_csv(output_path, index=False)
            
            logger.info(f"Custom data processed and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing custom data: {e}")
            raise

    def validate_labels(self, df: pd.DataFrame, label_column: str) -> bool:
        """
        Validate sentiment labels if provided.
        
        Args:
            df: Input DataFrame
            label_column: Name of the column containing sentiment labels
            
        Returns:
            bool: True if valid, raises ValueError if not
        """
        valid_labels = {'positive', 'negative', 'neutral'}
        invalid_labels = set(df[label_column].unique()) - valid_labels
        
        if invalid_labels:
            raise ValueError(
                f"Invalid sentiment labels found: {invalid_labels}. "
                f"Valid labels are: {valid_labels}"
            )
        return True

    def get_label_distribution(self, df: pd.DataFrame, label_column: str) -> Dict[str, int]:
        """
        Get distribution of sentiment labels.
        
        Args:
            df: Input DataFrame
            label_column: Name of the column containing sentiment labels
            
        Returns:
            Dict[str, int]: Distribution of labels
        """
        return df[label_column].value_counts().to_dict()

def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process custom CSV data file")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--text-column", type=str, default="text",
                      help="Name of the column containing text data")
    parser.add_argument("--label-column", type=str,
                      help="Optional: Name of the column containing sentiment labels")
    
    args = parser.parse_args()
    
    loader = CustomDataLoader()
    output_path = loader.load_and_process(
        Path(args.input_file),
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    print(f"Data processed successfully. Output saved to: {output_path}")

if __name__ == "__main__":
    main() 
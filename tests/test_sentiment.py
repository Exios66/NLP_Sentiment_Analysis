#!/usr/bin/env python3
"""
Test Suite for Sentiment Analysis Pipeline
Tests the core functionality of text processing and sentiment analysis.
"""

import unittest
from pathlib import Path
import sys

# Add parent directory to path for importing local modules
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.text_processor import TextProcessor
from src.models.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.text_processor = TextProcessor()
        cls.sentiment_analyzer = SentimentAnalyzer()
        
        # Test texts with known sentiments
        cls.test_texts = {
            'positive': "I'm feeling much better today after starting therapy. It's really helping me cope.",
            'negative': "I've been struggling with depression and anxiety. Everything feels overwhelming.",
            'neutral': "I went to my therapy appointment today. We discussed various coping mechanisms."
        }

    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        for sentiment, text in self.test_texts.items():
            processed = self.text_processor.process_text(text)
            
            # Check basic preprocessing results
            self.assertIsInstance(processed, dict)
            self.assertIn('cleaned_text', processed)
            self.assertIn('tokens', processed)
            self.assertIn('entities', processed)
            
            # Check that cleaned text is not empty
            self.assertTrue(len(processed['cleaned_text']) > 0)
            
            # Check that tokens were generated
            self.assertTrue(len(processed['tokens']) > 0)

    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        for expected_sentiment, text in self.test_texts.items():
            # Process and analyze text
            processed = self.text_processor.process_text(text)
            result = self.sentiment_analyzer.analyze_text(processed['cleaned_text'])
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('vader', result)
            self.assertIn('bert', result)
            self.assertIn('combined_sentiment', result)
            
            # Check VADER scores
            vader_scores = result['vader']
            self.assertIn('compound', vader_scores)
            self.assertIn('positive', vader_scores)
            self.assertIn('negative', vader_scores)
            self.assertIn('neutral', vader_scores)
            
            # Check score ranges
            self.assertTrue(-1 <= vader_scores['compound'] <= 1)
            self.assertTrue(0 <= vader_scores['positive'] <= 1)
            self.assertTrue(0 <= vader_scores['negative'] <= 1)
            self.assertTrue(0 <= vader_scores['neutral'] <= 1)
            
            # Check BERT scores
            bert_scores = result['bert']
            self.assertIn('positive', bert_scores)
            self.assertIn('negative', bert_scores)
            self.assertIn('neutral', bert_scores)
            
            # Check that probabilities sum to approximately 1
            bert_total = sum(bert_scores.values())
            self.assertAlmostEqual(bert_total, 1.0, places=2)
            
            # Check that the sentiment matches expected for strong examples
            if expected_sentiment in ['positive', 'negative']:
                self.assertEqual(result['combined_sentiment'], expected_sentiment)

    def test_empty_input(self):
        """Test handling of empty input."""
        # Test empty string
        processed = self.text_processor.process_text("")
        self.assertEqual(processed['cleaned_text'], "")
        
        # Test None
        processed = self.text_processor.process_text(None)
        self.assertEqual(processed['cleaned_text'], "")
        
        # Test whitespace
        processed = self.text_processor.process_text("   ")
        self.assertEqual(processed['cleaned_text'], "")

    def test_special_characters(self):
        """Test handling of special characters and URLs."""
        text = "Check out this link: https://example.com! 😊 #mentalhealth"
        processed = self.text_processor.process_text(text)
        
        # URLs should be removed if configured
        self.assertNotIn("https://", processed['cleaned_text'])
        
        # Emojis should be removed if configured
        self.assertNotIn("😊", processed['cleaned_text'])
        
        # Special characters should be handled
        self.assertNotIn("#", processed['cleaned_text'])

if __name__ == '__main__':
    unittest.main() 
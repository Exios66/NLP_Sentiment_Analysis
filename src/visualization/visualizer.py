#!/usr/bin/env python3
"""
Visualization Module
Creates various visualizations for sentiment analysis results
including word clouds, trend analysis, and statistical plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentVisualizer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the visualizer with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = Path("data/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.config['plots']['style'])
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['visualization']
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def create_wordcloud(self, texts: List[str], sentiment: Optional[str] = None) -> None:
        """Create and save word cloud visualization."""
        try:
            # Combine all texts
            text = ' '.join(texts)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=self.config['wordcloud']['width'],
                height=self.config['wordcloud']['height'],
                background_color=self.config['wordcloud']['background_color'],
                max_words=200,
                colormap='viridis'
            ).generate(text)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Save
            sentiment_str = f"_{sentiment}" if sentiment else ""
            plt.savefig(self.output_dir / f"wordcloud{sentiment_str}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            raise

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> None:
        """Create and save sentiment distribution visualization."""
        try:
            # Create sentiment distribution plot
            plt.figure(figsize=self.config['plots']['figure_size'])
            
            sentiment_counts = df['sentiment'].value_counts()
            colors = [self.colors[s] for s in sentiment_counts.index]
            
            # Create bar plot
            ax = sentiment_counts.plot(kind='bar', color=colors)
            plt.title('Distribution of Sentiments')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            
            # Add percentage labels
            total = sentiment_counts.sum()
            for i, v in enumerate(sentiment_counts):
                percentage = (v / total) * 100
                ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "sentiment_distribution.png", dpi=self.config['plots']['dpi'])
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating sentiment distribution plot: {e}")
            raise

    def plot_sentiment_trends(self, df: pd.DataFrame) -> None:
        """Create and save sentiment trends over time visualization."""
        try:
            # Convert timestamp to datetime
            df['created_utc'] = pd.to_datetime(df['created_utc'])
            
            # Group by date and sentiment
            daily_sentiments = df.groupby([
                df['created_utc'].dt.date,
                'sentiment'
            ]).size().unstack(fill_value=0)
            
            # Calculate percentages
            daily_percentages = daily_sentiments.div(daily_sentiments.sum(axis=1), axis=0) * 100
            
            # Create interactive plot with plotly
            fig = go.Figure()
            
            for sentiment in ['positive', 'neutral', 'negative']:
                fig.add_trace(go.Scatter(
                    x=daily_percentages.index,
                    y=daily_percentages[sentiment],
                    name=sentiment.capitalize(),
                    line=dict(color=self.colors[sentiment]),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Sentiment Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Percentage',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            fig.write_html(self.output_dir / "sentiment_trends.html")
            
        except Exception as e:
            logger.error(f"Error creating sentiment trends plot: {e}")
            raise

    def create_subreddit_analysis(self, df: pd.DataFrame) -> None:
        """Create and save subreddit-specific analysis visualizations."""
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sentiment Distribution by Subreddit',
                              'Average Sentiment Scores by Subreddit'),
                vertical_spacing=0.3
            )
            
            # Sentiment distribution by subreddit
            subreddit_sentiments = pd.crosstab(df['subreddit'], df['sentiment'])
            subreddit_sentiments_pct = subreddit_sentiments.div(
                subreddit_sentiments.sum(axis=1), axis=0
            ) * 100
            
            # Add stacked bar chart
            for sentiment in ['positive', 'neutral', 'negative']:
                fig.add_trace(
                    go.Bar(
                        name=sentiment.capitalize(),
                        x=subreddit_sentiments_pct.index,
                        y=subreddit_sentiments_pct[sentiment],
                        marker_color=self.colors[sentiment]
                    ),
                    row=1, col=1
                )
            
            # Average sentiment scores by subreddit
            avg_scores = df.groupby('subreddit').agg({
                'sentiment_scores': lambda x: pd.eval(str(list(x)))[0]['vader']['compound'].mean()
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=avg_scores['subreddit'],
                    y=avg_scores['sentiment_scores'],
                    marker_color='#3498db'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                barmode='stack',
                title_text="Subreddit Sentiment Analysis"
            )
            
            fig.write_html(self.output_dir / "subreddit_analysis.html")
            
        except Exception as e:
            logger.error(f"Error creating subreddit analysis: {e}")
            raise

    def create_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Create and save correlation matrix of numerical features."""
        try:
            # Extract numerical features
            numerical_features = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numerical_features].corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f'
            )
            
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=self.config['plots']['dpi'])
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            raise

    def generate_report(self, df: pd.DataFrame) -> None:
        """Generate a comprehensive HTML report with all visualizations."""
        try:
            # Create report directory
            report_dir = self.output_dir / "report"
            report_dir.mkdir(exist_ok=True)
            
            # Basic statistics
            total_posts = len(df)
            sentiment_stats = df['sentiment'].value_counts().to_dict()
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Sentiment Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 1200px; margin: auto; }}
                    .section {{ margin-bottom: 40px; }}
                    .stat-box {{ 
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Sentiment Analysis Report</h1>
                    <div class="section">
                        <h2>Overview</h2>
                        <div class="stat-box">
                            <p>Total Posts Analyzed: {total_posts}</p>
                            <p>Sentiment Distribution:</p>
                            <ul>
                                <li>Positive: {sentiment_stats.get('positive', 0)} ({sentiment_stats.get('positive', 0)/total_posts*100:.1f}%)</li>
                                <li>Neutral: {sentiment_stats.get('neutral', 0)} ({sentiment_stats.get('neutral', 0)/total_posts*100:.1f}%)</li>
                                <li>Negative: {sentiment_stats.get('negative', 0)} ({sentiment_stats.get('negative', 0)/total_posts*100:.1f}%)</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Visualizations</h2>
                        <h3>Word Clouds</h3>
                        <img src="wordcloud.png" alt="Overall Word Cloud" style="max-width: 100%;">
                        
                        <h3>Sentiment Distribution</h3>
                        <img src="sentiment_distribution.png" alt="Sentiment Distribution" style="max-width: 100%;">
                        
                        <h3>Sentiment Trends</h3>
                        <iframe src="sentiment_trends.html" width="100%" height="600px" frameborder="0"></iframe>
                        
                        <h3>Subreddit Analysis</h3>
                        <iframe src="subreddit_analysis.html" width="100%" height="800px" frameborder="0"></iframe>
                        
                        <h3>Correlation Matrix</h3>
                        <img src="correlation_matrix.png" alt="Correlation Matrix" style="max-width: 100%;">
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save report
            with open(report_dir / "report.html", 'w') as f:
                f.write(html_content)
            
            logger.info(f"Report generated at {report_dir / 'report.html'}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def create_all_visualizations(self, input_path: Path) -> None:
        """Create all visualizations from input data."""
        try:
            # Read input data
            df = pd.read_csv(input_path)
            
            # Create visualizations
            logger.info("Creating word cloud...")
            self.create_wordcloud(df['cleaned_text'].tolist())
            
            logger.info("Creating sentiment distribution plot...")
            self.plot_sentiment_distribution(df)
            
            logger.info("Creating sentiment trends plot...")
            self.plot_sentiment_trends(df)
            
            logger.info("Creating subreddit analysis...")
            self.create_subreddit_analysis(df)
            
            logger.info("Creating correlation matrix...")
            self.create_correlation_matrix(df)
            
            logger.info("Generating comprehensive report...")
            self.generate_report(df)
            
            logger.info("All visualizations created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

def main():
    """Main function to run the visualizer."""
    try:
        visualizer = SentimentVisualizer()
        
        # Find the latest sentiment analysis results
        analyzed_dir = Path("data/processed/sentiment")
        latest_file = max(analyzed_dir.glob("sentiment_*.csv"), key=lambda x: x.stat().st_mtime)
        
        # Create visualizations
        visualizer.create_all_visualizations(latest_file)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main() 
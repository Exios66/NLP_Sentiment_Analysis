#!/usr/bin/env python3
"""
Reddit Data Collector for Mental Health Subreddits
This script collects posts and comments from mental health-related subreddits.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import praw
import pandas as pd
import yaml
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditCollector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Reddit collector with configuration."""
        load_dotenv()
        self.config = self._load_config(config_path)
        self.reddit = self._initialize_reddit()
        self.output_dir = Path("data/raw/reddit")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['data_collection']['reddit']
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize Reddit API connection."""
        try:
            return praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'Mental Health Analysis Bot 1.0')
            )
        except Exception as e:
            logger.error(f"Error initializing Reddit API: {e}")
            raise

    def collect_posts(self, subreddit_name: str) -> List[Dict[str, Any]]:
        """Collect posts from a subreddit."""
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in tqdm(
                subreddit.top(time_filter=self.config['time_filter'], 
                             limit=self.config['post_limit']),
                desc=f"Collecting posts from r/{subreddit_name}"
            ):
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit_name,
                    'comments': self._get_comments(post)
                }
                posts.append(post_data)
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
        return posts

    def _get_comments(self, post: praw.models.Submission) -> List[Dict[str, Any]]:
        """Collect comments from a post."""
        comments = []
        try:
            post.comments.replace_more(limit=0)
            for comment in post.comments.list()[:self.config['comment_limit']]:
                comment_data = {
                    'comment_id': comment.id,
                    'text': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    'score': comment.score,
                    'is_submitter': comment.is_submitter
                }
                comments.append(comment_data)
        except Exception as e:
            logger.error(f"Error collecting comments from post {post.id}: {e}")
        return comments

    def run_collection(self):
        """Run the collection process for all configured subreddits."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_data = []

        for subreddit in self.config['subreddits']:
            logger.info(f"Collecting data from r/{subreddit}")
            posts = self.collect_posts(subreddit)
            all_data.extend(posts)

            # Save individual subreddit data
            output_file = self.output_dir / f"{subreddit}_{timestamp}.json"
            self._save_data(posts, output_file)

        # Save combined data
        combined_file = self.output_dir / f"combined_{timestamp}.json"
        self._save_data(all_data, combined_file)

        # Create DataFrame for analysis
        df = self._create_dataframe(all_data)
        df.to_csv(self.output_dir / f"reddit_data_{timestamp}.csv", index=False)

    def _save_data(self, data: List[Dict[str, Any]], output_file: Path):
        """Save collected data to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving data to {output_file}: {e}")

    def _create_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a pandas DataFrame from collected data."""
        posts_data = []
        comments_data = []

        for post in data:
            post_data = {k: v for k, v in post.items() if k != 'comments'}
            posts_data.append(post_data)

            for comment in post['comments']:
                comment_data = {
                    'post_id': post['post_id'],
                    'subreddit': post['subreddit'],
                    **comment
                }
                comments_data.append(comment_data)

        posts_df = pd.DataFrame(posts_data)
        comments_df = pd.DataFrame(comments_data)

        return pd.concat([posts_df, comments_df], axis=0, ignore_index=True)

if __name__ == "__main__":
    try:
        collector = RedditCollector()
        collector.run_collection()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise 
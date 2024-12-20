# Mental Health Text Sentiment Analysis Pipeline

## Overview

This project implements a comprehensive Natural Language Processing (NLP) pipeline for analyzing sentiment in mental health-related text data. It includes tools for data collection, preprocessing, sentiment analysis, and visualization of mental health discussions.

## Features

- Data Collection: Reddit API integration for mental health subreddits
- Text Preprocessing: Advanced NLP preprocessing using NLTK and spaCy
- Sentiment Analysis: Multiple models including VADER and BERT-based approaches
- Visualization: Interactive dashboards and statistical analysis
- Data Storage: Efficient data management and caching
- API: RESTful API for model serving

## Project Structure

```
├── data/                      # Data storage
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Preprocessed data
│   └── models/              # Trained models
├── src/                      # Source code
│   ├── data_collection/     # Scripts for data collection
│   ├── preprocessing/       # Text preprocessing modules
│   ├── models/             # Model implementation
│   ├── visualization/      # Visualization tools
│   └── api/                # API implementation
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/NLP_Sentiment_Analysis.git
cd NLP_Sentiment_Analysis
```

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

1. Data Collection:

```bash
python src/data_collection/reddit_scraper.py
```

1. Preprocessing:

```bash
python src/preprocessing/preprocess.py
```

3. Model Training:

```bash
python src/models/train.py
```

4. Run Analysis:

```bash
python src/visualization/analyze.py
```

5. Start API:

```bash
python src/api/app.py
```

## Configuration

- Configure data sources in `config/data_sources.yaml`
- Model parameters in `config/model_config.yaml`
- API settings in `config/api_config.yaml`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Your Name

## Acknowledgments

- NLTK
- spaCy
- Transformers
- PRAW (Python Reddit API Wrapper)

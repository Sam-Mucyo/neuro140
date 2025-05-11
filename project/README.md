# LLM-Based Topic Modeling and Analysis

This project explores topic modeling techniques on mental health-related Reddit data, comparing traditional LDA approaches with LLM-structured topic modeling.

## Environment Setup

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib scikit-learn
pip install nltk tqdm python-dotenv
pip install openai tiktoken
pip install plotly networkx tabulate seaborn wordcloud
pip install jupyterlab ipython

# Note: Some packages like gensim and spacy may require additional system dependencies
# If you encounter build errors, consider using conda (Option 2)
```

### Option 2: Using Conda

The repository includes a `neuro140.yml` file for creating a Conda environment:

```bash
# Create and activate the environment
conda env create -f neuro140.yml
conda activate neuro140

# Install spaCy language model
python -m spacy download en_core_web_sm
```

## OpenAI API Setup

This project uses OpenAI's API for LLM-based topic modeling. Create an `.env` file based on the provided `.env.example` with your API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Data

The `data/` directory contains TF-IDF feature vectors extracted from various mental health-related subreddits, with files organized by subreddit, time period (pre/post COVID-19), and year.

## Running the Analysis

- `eda.ipynb`: Exploratory data analysis notebook
- `traditional_lda.ipynb`: Traditional LDA topic modeling
- `comparative_analysis.ipynb`: Comparison of traditional and LLM-based topic modeling
- `pipeline.py`: End-to-end topic modeling pipeline
- `llm_structured_topic_modeling.py`: Implementation of LLM-based structured topic modeling

## Evaluation & Visualization

- `topic_model_evaluation.py`: Tools for evaluating topic model quality
- `topic_model_visualization.py`: Visualization utilities for topic model outputs

## Results

The `results/` directory contains saved model outputs and evaluation metrics.
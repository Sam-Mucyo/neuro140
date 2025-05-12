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

The `data/` directory contains TF-IDF feature vectors extracted from various mental health-related subreddits, with files organized by subreddit, time period (pre/post COVID-19), and year. You can download the dataset from [here](https://zenodo.org/records/3941387) (we gitignored it for size reasons).

## Running the Analysis

### Notebooks (in `experiments/`)

- `experiments/eda.ipynb`: Exploratory data analysis notebook
- `experiments/traditional_lda.ipynb`: Traditional LDA topic modeling
- `experiments/comparative_analysis.ipynb`: Comparison of traditional and LLM-based topic modeling and evaluation

### Scripts (in `src/`)

- `src/pipeline.py`: End-to-end topic modeling pipeline (pilot + full run)
- `src/llm_structured_topic_modeling.py`: LLM-based structured topic extraction pipeline
- `src/estimate_tokens.py`: Token and cost estimation utility for OpenAI models
- `src/topic_model_evaluation.py`: Functions for computing topic model metrics
- `src/topic_model_visualization.py`: Utilities for plotting topics and metrics

## Evaluation & Visualization

- `topic_model_evaluation.py`: Tools for evaluating topic model quality
- `topic_model_visualization.py`: Visualization utilities for topic model outputs

## Results

The `results/` directory contains saved model outputs and evaluation metrics.

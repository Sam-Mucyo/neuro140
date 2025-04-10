#!/usr/bin/env python3
"""
LLM-Based Structurally Focused Topic Modeling

This script implements the pipeline, which uses LLMs to extract
structured representations from mental health forum posts before applying LDA
for topic modeling.

What to Expect When Running

The script will first validate if your OpenAI API key is available.

It will then run a pilot test on 5 examples from the dataset, showing:
    The original text
    The structured data extracted by the LLM
    The tokens used for topic modeling

After the pilot test, you'll be asked if you want to proceed with the full pipeline.

If you choose to proceed, you can specify a sample size (for testing with a subset of data) or use all available data.

The script will:
    Process all selected posts through the LLM
    Run topic modeling on both the structured and baseline texts
    Compare coherence scores
    Save the models and results to the results directory
    Key Implementation Details

The script follows the methodology described in your final report:
    LLM Prompting Strategy: The prompt is designed to extract specific mental health-related fields from the text.
    Structured Text Creation: The extracted fields are combined into a structured text representation.
    Topic Modeling: LDA is applied to both the structured text and the original text for comparison.
    Evaluation: Topics are evaluated using coherence scores and representative documents.



"""

import os
import csv
import pandas as pd
import numpy as np
import re
import json
import time
import logging
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel
import spacy
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configuration
DATA_PATH = "data/mentalhealth_post_features_tfidf_256.csv"
RESULTS_DIR = "results"
PILOT_SAMPLE_SIZE = 5

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory: {RESULTS_DIR}")

# Download NLTK resources if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load spaCy for lemmatization
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load English stopwords
stop_words = set(stopwords.words("english"))
# Add domain-specific stopwords
mental_health_stopwords = {
    "feel",
    "feeling",
    "felt",
    "just",
    "like",
    "know",
    "think",
    "get",
    "got",
    "really",
}
stop_words = stop_words.union(mental_health_stopwords)


def load_data(data_path=DATA_PATH):
    """Load data from CSV file."""
    print(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def preprocess_text(text):
    """Basic preprocessing for text."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text):
    """Lemmatize text using spaCy."""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_.lower() not in stop_words
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    return tokens


def extract_structured_data_with_llm(text, model="gpt-3.5-turbo"):
    """
    Extract structured data from text using OpenAI's GPT model.

    Args:
        text: The text to analyze
        model: The OpenAI model to use

    Returns:
        Dictionary containing structured data
    """
    if not OPENAI_API_KEY:
        print("Warning: OpenAI API key not found. Returning empty structure.")
        return {
            "main_issue": "",
            "emotional_tone": "",
            "cognitive_stress_markers": "",
            "suicidal_red_flags": "",
            "keywords": [],
        }

    # Truncate text if it's too long (to save tokens)
    max_length = 4000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # Define the prompt for structured extraction
    prompt = f"""
    Read the following mental health forum post carefully. Then provide a JSON with the following fields:
    
    1. main_issue: The primary mental health concern or problem discussed
    2. emotional_tone: The dominant emotions expressed (e.g., anxiety, depression, anger)
    3. cognitive_stress_markers: Any cognitive distortions or stress indicators
    4. suicidal_red_flags: Any explicit or implicit references to self-harm or suicide
    5. keywords: A list of 5-10 key terms that best represent the content
    
    Post: {text}
    
    JSON response:
    """

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mental health text analyzer that extracts structured information from forum posts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        # Extract the JSON response
        json_str = response.choices[0].message.content

        # Clean up the JSON string if needed
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        # Parse the JSON
        structured_data = json.loads(json_str)
        return structured_data

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "main_issue": "",
            "emotional_tone": "",
            "cognitive_stress_markers": "",
            "suicidal_red_flags": "",
            "keywords": [],
        }


def create_structured_text(structured_data):
    """
    Create a structured text representation from the extracted data.

    Args:
        structured_data: Dictionary containing structured data

    Returns:
        String containing structured text
    """
    structured_text = ""

    # Add main issue
    if structured_data.get("main_issue"):
        structured_text += f"ISSUE: {structured_data['main_issue']} "

    # Add emotional tone
    if structured_data.get("emotional_tone"):
        structured_text += f"EMOTION: {structured_data['emotional_tone']} "

    # Add cognitive stress markers
    if structured_data.get("cognitive_stress_markers"):
        structured_text += f"COGNITIVE: {structured_data['cognitive_stress_markers']} "

    # Add suicidal red flags
    if structured_data.get("suicidal_red_flags"):
        structured_text += f"SUICIDE_RISK: {structured_data['suicidal_red_flags']} "

    # Add keywords
    if structured_data.get("keywords") and isinstance(
        structured_data["keywords"], list
    ):
        keywords = " ".join(structured_data["keywords"])
        structured_text += f"KEYWORDS: {keywords}"

    return structured_text


def process_batch(df, use_llm=True):
    """
    Process a batch of data.

    Args:
        df: DataFrame containing the batch to process
        use_llm: Whether to use LLM for structured extraction

    Returns:
        DataFrame with processed data
    """
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing posts"):
        # Extract text from row
        text = row.get("post", "")

        # Skip empty text
        if not text or not isinstance(text, str):
            continue

        result = {
            "id": row.get("id", idx),
            "original_text": text,
            "processed_text": preprocess_text(text),
        }

        # Use LLM for structured extraction if requested
        if use_llm:
            structured_data = extract_structured_data_with_llm(text)
            structured_text = create_structured_text(structured_data)
            result["structured_data"] = structured_data
            result["structured_text"] = structured_text
            # Create tokens from structured text
            result["structured_tokens"] = lemmatize_text(structured_text)

        # Create tokens from original text (for baseline comparison)
        result["baseline_tokens"] = lemmatize_text(result["processed_text"])

        # Add to results
        results.append(result)

    return pd.DataFrame(results)


def build_lda_model(texts, num_topics=10, passes=20):
    """
    Build an LDA model from texts.

    Args:
        texts: List of tokenized texts
        num_topics: Number of topics to extract
        passes: Number of passes through the corpus during training

    Returns:
        Tuple of (dictionary, corpus, lda_model)
    """
    # Create a dictionary
    dictionary = corpora.Dictionary(texts)

    # Filter out extremes
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Create a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Build the LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha="auto",
        eta="auto",
    )

    return dictionary, corpus, lda_model


def evaluate_topics(lda_model, corpus, dictionary, texts):
    """
    Evaluate topic coherence.

    Args:
        lda_model: Trained LDA model
        corpus: Document-term matrix
        dictionary: Dictionary mapping terms to IDs
        texts: List of tokenized texts

    Returns:
        Coherence score
    """
    # Calculate coherence score
    coherence_model = gensim.models.CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence="c_v"
    )

    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

    return coherence_score


def find_optimal_topics(texts, start=5, limit=20, step=5):
    """
    Find the optimal number of topics.

    Args:
        texts: List of tokenized texts
        start: Starting number of topics
        limit: Maximum number of topics
        step: Step size for topic number

    Returns:
        Optimal number of topics
    """
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit + 1, step):
        print(f"Building LDA model with {num_topics} topics...")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=20,
            alpha="auto",
            eta="auto",
        )

        model_list.append(model)

        coherencemodel = gensim.models.CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )

        coherence_values.append(coherencemodel.get_coherence())

    # Find the optimal number of topics
    optimal_idx = coherence_values.index(max(coherence_values))
    optimal_num_topics = range(start, limit + 1, step)[optimal_idx]

    print(f"Optimal number of topics: {optimal_num_topics}")

    return optimal_num_topics


def get_representative_docs(data, lda_model, corpus, n_docs=3):
    """
    Get representative documents for each topic.

    Args:
        data: DataFrame containing the data
        lda_model: Trained LDA model
        corpus: Document-term matrix
        n_docs: Number of representative documents to return

    Returns:
        Dictionary mapping topic IDs to representative documents
    """
    # Get the dominant topic for each document
    document_topics = []
    for i, doc_topics in enumerate(lda_model[corpus]):
        # Sort topics by probability
        sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        # Get the dominant topic
        if sorted_topics:
            dominant_topic = sorted_topics[0][0]
            topic_prob = sorted_topics[0][1]
        else:
            dominant_topic = -1
            topic_prob = 0
        document_topics.append((dominant_topic, topic_prob))

    # Add dominant topics to data
    data_copy = data.copy()
    data_copy["dominant_topic"] = [topic for topic, _ in document_topics]
    data_copy["topic_probability"] = [prob for _, prob in document_topics]

    # Get representative documents
    topic_docs = {}
    for topic_id in range(lda_model.num_topics):
        # Filter documents for this topic and sort by probability
        topic_data = data_copy[data_copy["dominant_topic"] == topic_id].sort_values(
            "topic_probability", ascending=False
        )
        # Get top n documents
        if len(topic_data) > 0:
            top_docs = topic_data["original_text"].head(n_docs).tolist()
            topic_docs[topic_id] = top_docs
        else:
            topic_docs[topic_id] = []

    return topic_docs


def run_pilot(df=None):
    """
    Run a pilot test on a small sample of the data.

    Args:
        df: DataFrame containing the data (if None, will be loaded)

    Returns:
        Results of the pilot test
    """
    print(f"\n🧪 Running pilot test on {PILOT_SAMPLE_SIZE} records")

    # Load data if not provided
    if df is None:
        df = load_data()

    # Take a small sample
    sample_df = df.head(PILOT_SAMPLE_SIZE)

    # Process the sample
    results_df = process_batch(sample_df)

    print("\nPilot test results sample:")
    for i, row in results_df.iterrows():
        print(f"\nRecord {i+1}:")
        print(f"Original text (truncated): {row['original_text'][:100]}...")

        if "structured_data" in row:
            print(f"Structured data: {json.dumps(row['structured_data'], indent=2)}")
            print(f"Structured text: {row['structured_text']}")
            print(f"Structured tokens: {row['structured_tokens'][:10]}...")

        print(f"Baseline tokens: {row['baseline_tokens'][:10]}...")

    # Save pilot results
    results_df.to_csv(os.path.join(RESULTS_DIR, "pilot_results.csv"), index=False)
    print(
        f"Saved {len(results_df)} records to {os.path.join(RESULTS_DIR, 'pilot_results.csv')}"
    )

    return results_df


def run_topic_modeling(processed_df, use_structured=True, num_topics=None):
    """
    Run topic modeling on processed data.

    Args:
        processed_df: DataFrame containing processed data
        use_structured: Whether to use structured text for topic modeling
        num_topics: Number of topics (if None, will be determined automatically)

    Returns:
        Tuple of (dictionary, corpus, lda_model)
    """
    # Select the appropriate tokens
    if use_structured and "structured_tokens" in processed_df.columns:
        print("Using structured tokens for topic modeling")
        texts = processed_df["structured_tokens"].tolist()
        text_type = "structured"
    else:
        print("Using baseline tokens for topic modeling")
        texts = processed_df["baseline_tokens"].tolist()
        text_type = "baseline"

    # Remove empty token lists
    texts = [tokens for tokens in texts if tokens]

    # Find optimal number of topics if not provided
    if num_topics is None:
        num_topics = find_optimal_topics(texts)

    # Build the LDA model
    dictionary, corpus, lda_model = build_lda_model(texts, num_topics=num_topics)

    # Evaluate the model
    coherence_score = evaluate_topics(lda_model, corpus, dictionary, texts)

    # Print the topics
    print("\nTopics:")
    for topic_id in range(lda_model.num_topics):
        print(f"Topic {topic_id}: {lda_model.print_topic(topic_id)}")

    # Get representative documents
    representative_docs = get_representative_docs(processed_df, lda_model, corpus)

    # Print representative documents
    print("\nRepresentative Documents:")
    for topic_id, docs in representative_docs.items():
        print(f"\nTopic {topic_id} - Top Words: {lda_model.print_topic(topic_id)}")
        print("Representative Documents:")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}: {doc[:200]}...")

    # Save the model and results
    model_dir = os.path.join(RESULTS_DIR, f"lda_model_{text_type}_{num_topics}_topics")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    lda_model.save(os.path.join(model_dir, "lda_model"))

    # Save the topics
    with open(os.path.join(model_dir, "topics.txt"), "w") as f:
        for topic_id in range(lda_model.num_topics):
            f.write(f"Topic {topic_id}: {lda_model.print_topic(topic_id)}\n")

    # Save the coherence score
    with open(os.path.join(model_dir, "coherence.txt"), "w") as f:
        f.write(f"Coherence Score: {coherence_score}\n")

    return dictionary, corpus, lda_model


def run_full_pipeline(df=None, sample_size=None):
    """
    Run the full pipeline on all data or a sample.

    Args:
        df: DataFrame containing the data (if None, will be loaded)
        sample_size: Size of the sample to use (if None, use all data)

    Returns:
        Results from processing all data
    """
    print("\n🚀 Running full pipeline")
    start_time = time.time()

    # Load data if not provided
    if df is None:
        df = load_data()

    # Take a sample if requested
    if sample_size is not None:
        print(f"Using a sample of {sample_size} records")
        df = df.sample(sample_size, random_state=42)

    # Process all data
    processed_df = process_batch(df)

    # Save processed data
    processed_df.to_csv(os.path.join(RESULTS_DIR, "processed_data.csv"), index=False)
    print(
        f"Saved {len(processed_df)} processed records to {os.path.join(RESULTS_DIR, 'processed_data.csv')}"
    )

    # Run topic modeling with structured text
    print("\nRunning topic modeling with structured text:")
    structured_results = run_topic_modeling(processed_df, use_structured=True)

    # Run topic modeling with baseline text (for comparison)
    print("\nRunning topic modeling with baseline text (for comparison):")
    baseline_results = run_topic_modeling(processed_df, use_structured=False)

    # Calculate and log total runtime
    total_time = time.time() - start_time
    print(f"\n✅ Full pipeline completed with {len(processed_df)} results")
    print(f"Total runtime: {total_time:.2f} seconds")

    return processed_df, structured_results, baseline_results


def main():
    """Main entry point for the pipeline."""
    # Validate API keys
    if not OPENAI_API_KEY:
        print("⚠️ Warning: OpenAI API key not found in environment variables.")
        print(
            "Please set the OPENAI_API_KEY environment variable or add it to a .env file."
        )
    else:
        print("✅ OpenAI API key loaded successfully.")

    # Run the pilot test
    pilot_df = run_pilot()

    # Ask for confirmation before running the full pipeline
    user_input = input("\nProceed with full pipeline run? (y/n): ")

    if user_input.lower() == "y":
        # Run the full pipeline with a limited sample for demonstration
        sample_size = int(input("Enter sample size (or 0 for all data): "))
        if sample_size <= 0:
            sample_size = None

        run_full_pipeline(sample_size=sample_size)
    else:
        print("Full pipeline run cancelled")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LLM-Based Structurally Focused Topic Modeling

This script implements the pipeline, which uses LLMs to extract
structured representations from mental health forum posts before applying LDA
for topic modeling.

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

# Constants
DATA_PATH = "../data/mentalhealth_post_features_tfidf_256.csv"  # Path to the data file
RESULTS_DIR = "../results"  # Directory to store results
PILOT_SAMPLE_SIZE = 5  # Number of records to use for pilot testing
RANDOM_SEED = 42  # Random seed for reproducibility

# Model selection
DEFAULT_MODEL = "gpt-4o-mini"  # Default model to use (supports Structured Outputs)

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


def analyze_text(text):
    """
    Analyze a single text entry and return structured results using OpenAI's Structured Outputs feature.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with structured analysis results
    """
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define the system prompt
    system_prompt = """You are an AI assistant trained to analyze mental health text data.
    Extract key information from the provided text and categorize it according to the schema."""

    try:
        # Define the schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "description": "Main themes or topics identified in the text",
                    "items": {"type": "string"},
                },
                "emotional_tone": {
                    "type": "string",
                    "description": "Overall emotional tone of the text",
                    "enum": ["positive", "negative", "neutral", "mixed", "unknown"],
                },
                "concerns": {
                    "type": "array",
                    "description": "Key concerns or issues mentioned in the text",
                    "items": {"type": "string"},
                },
                "cognitive_patterns": {
                    "type": "array",
                    "description": "Any cognitive patterns or distortions identified in the text",
                    "items": {"type": "string"},
                },
                "social_context": {
                    "type": "array",
                    "description": "Social context or relationships mentioned in the text",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "themes",
                "emotional_tone",
                "concerns",
                "cognitive_patterns",
                "social_context",
            ],
            "additionalProperties": False,
        }

        # Make the API call using the new Responses API with Structured Outputs
        response = client.responses.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency with Structured Outputs support
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "mental_health_analysis",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        # Extract the structured result
        structured_result = json.loads(response.output_text)

        # Add the original text to the result
        structured_result["original_text"] = text

        return structured_result

    except Exception as e:
        print(f"Error analyzing text: {e}")
        # Return a minimal result in case of error
        return {
            "themes": ["Error in analysis"],
            "emotional_tone": "unknown",
            "concerns": [],
            "cognitive_patterns": [],
            "social_context": [],
            "original_text": text,
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
    if structured_data.get("themes"):
        structured_text += f"THEMES: {', '.join(structured_data['themes'])} "

    # Add emotional tone
    if structured_data.get("emotional_tone"):
        structured_text += f"EMOTION: {structured_data['emotional_tone']} "

    # Add cognitive stress markers
    if structured_data.get("cognitive_patterns"):
        structured_text += (
            f"COGNITIVE: {', '.join(structured_data['cognitive_patterns'])} "
        )

    # Add suicidal red flags
    if structured_data.get("concerns"):
        structured_text += f"CONCERNS: {', '.join(structured_data['concerns'])} "

    # Add keywords
    if structured_data.get("social_context"):
        keywords = ", ".join(structured_data["social_context"])
        structured_text += f"KEYWORDS: {keywords}"

    return structured_text


def process_batch(df, model=DEFAULT_MODEL, checkpoint_interval=50):
    """
    Process a batch of text entries.

    Args:
        df: DataFrame containing the text data
        model: Model to use for analysis (must support Structured Outputs)
        checkpoint_interval: Save progress after processing this many records

    Returns:
        DataFrame with processed results
    """
    results = []
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoints directory: {checkpoint_dir}")
    
    # Check if there are existing checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".csv")])
    
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Found checkpoint: {latest_checkpoint}")
        checkpoint_df = pd.read_csv(latest_checkpoint)
        results = checkpoint_df.to_dict('records')
        processed_ids = set(checkpoint_df["id"].astype(str))
        print(f"Resuming from checkpoint with {len(results)} already processed records")
    else:
        processed_ids = set()
        print("No checkpoints found, starting from scratch")

    # Process each text entry
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        # Skip already processed records
        current_id = row["id"] if "id" in row else str(idx)
        if current_id in processed_ids:
            continue
            
        text = row["post"]
        result = analyze_text(text)

        # Add the index to the result
        result["id"] = current_id

        # Create structured text representation
        structured_text = create_structured_text(result)
        result["structured_text"] = structured_text

        # Create tokens from structured text
        result["structured_tokens"] = lemmatize_text(structured_text)

        # Create tokens from original text (for baseline comparison)
        result["processed_text"] = preprocess_text(text)
        result["baseline_tokens"] = lemmatize_text(result["processed_text"])

        # Add to results
        results.append(result)
        processed_ids.add(current_id)
        
        # Save checkpoint periodically
        if len(results) % checkpoint_interval == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{len(results)}.csv")
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(checkpoint_file, index=False)
            print(f"Saved checkpoint with {len(results)} records to {checkpoint_file}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final checkpoint
    final_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_final_{len(results)}.csv")
    results_df.to_csv(final_checkpoint, index=False)
    print(f"Saved final checkpoint with {len(results_df)} records to {final_checkpoint}")

    return results_df


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


def run_pilot(df=None, skip_existing=True):
    """
    Run a pilot test on a small sample of the data.

    Args:
        df: DataFrame containing the data (if None, will be loaded)
        skip_existing: If True, skip processing if pilot results already exist

    Returns:
        Results of the pilot test
    """
    print(f"\n🧪 Running pilot test on {PILOT_SAMPLE_SIZE} records")

    pilot_results_path = os.path.join(RESULTS_DIR, "pilot_results.csv")

    # Load data if not provided
    if df is None:
        df = load_data()

    # Take a deterministic sample using the random seed
    # Instead of head(), we use sample() with a fixed seed for deterministic selection
    sample_df = df.sample(PILOT_SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Ensure we have an 'id' column for tracking
    if "id" not in sample_df.columns:
        sample_df["id"] = sample_df.index.astype(str)

    # Check if pilot results already exist and we want to skip processing
    existing_results_df = None
    if skip_existing and os.path.exists(pilot_results_path):
        print(f"Pilot results exist at {pilot_results_path}")
        try:
            existing_results_df = pd.read_csv(pilot_results_path)
            print(f"Loaded {len(existing_results_df)} existing pilot results")

            # If we have all the posts we need already processed, return them
            if set(sample_df["id"].astype(str)).issubset(
                set(existing_results_df["id"].astype(str))
            ):
                print("All selected posts have already been processed")
                # Filter to only include the posts in our current sample
                filtered_results = existing_results_df[
                    existing_results_df["id"]
                    .astype(str)
                    .isin(sample_df["id"].astype(str))
                ]
                return filtered_results
            else:
                print("Some selected posts have not been processed yet")
        except Exception as e:
            print(f"Error loading existing pilot results: {e}")
            print("Will process all selected posts")
            existing_results_df = None

    # Process the sample, skipping any already processed posts
    if existing_results_df is not None:
        # Find posts that need processing
        already_processed_ids = set(existing_results_df["id"].astype(str))
        new_posts_df = sample_df[
            ~sample_df["id"].astype(str).isin(already_processed_ids)
        ]

        if len(new_posts_df) > 0:
            print(f"Processing {len(new_posts_df)} new posts")
            new_results_df = process_batch(new_posts_df)

            # Combine with existing results
            existing_to_keep = existing_results_df[
                existing_results_df["id"].astype(str).isin(sample_df["id"].astype(str))
            ]
            results_df = pd.concat(
                [existing_to_keep, new_results_df], ignore_index=True
            )
        else:
            print("No new posts to process")
            # Filter existing results to only include our current sample
            results_df = existing_results_df[
                existing_results_df["id"].astype(str).isin(sample_df["id"].astype(str))
            ]
    else:
        # Process all posts in the sample
        print(f"Processing all {len(sample_df)} selected posts")
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


def run_full_pipeline(df=None, skip_existing=True):
    """
    Run the full pipeline on all data or a sample.

    Args:
        df: DataFrame containing the data (if None, will be loaded)
        sample_size: Size of the sample to use (if None, use all data)
        skip_existing: If True, skip processing if results already exist

    Returns:
        Results from processing all data
    """
    print("\n🚀 Running full pipeline")
    start_time = time.time()

    # Define the path for processed data
    processed_data_path = os.path.join(RESULTS_DIR, "processed_data.csv")

    # Load data if not provided
    if df is None:
        df = load_data()

    # Ensure we have an 'id' column for tracking
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    # Check if processed data already exists and we want to skip processing
    existing_processed_df = None
    if skip_existing and os.path.exists(processed_data_path):
        print(f"Processed data exists at {processed_data_path}")
        try:
            existing_processed_df = pd.read_csv(processed_data_path)
            print(f"Loaded {len(existing_processed_df)} existing processed records")

            # Check if all the posts we need are already processed
            current_ids = set(df["id"].astype(str))
            existing_ids = set(existing_processed_df["id"].astype(str))

            if current_ids.issubset(existing_ids):
                print("All selected posts have already been processed")
                # Filter to only include the posts in our current selection
                filtered_results = existing_processed_df[
                    existing_processed_df["id"].astype(str).isin(df["id"].astype(str))
                ]

                # Run topic modeling on filtered data
                print("\nRunning topic modeling with existing processed data...")
                structured_results = run_topic_modeling(
                    filtered_results, use_structured=True
                )
                baseline_results = run_topic_modeling(
                    filtered_results, use_structured=False
                )

                return filtered_results, structured_results, baseline_results
        except Exception as e:
            print(f"Error loading existing processed data: {e}")
            print("Will process all selected posts")
            existing_processed_df = None

    # Process data, skipping any already processed posts
    if existing_processed_df is not None:
        # Find posts that need processing
        already_processed_ids = set(existing_processed_df["id"].astype(str))
        new_posts_df = df[~df["id"].astype(str).isin(already_processed_ids)]

        if len(new_posts_df) > 0:
            print(f"Processing {len(new_posts_df)} new posts")
            new_results_df = process_batch(new_posts_df)

            # Combine with existing results that are in our current selection
            existing_to_keep = existing_processed_df[
                existing_processed_df["id"].astype(str).isin(df["id"].astype(str))
            ]
            processed_df = pd.concat(
                [existing_to_keep, new_results_df], ignore_index=True
            )
        else:
            print("No new posts to process")
            # Filter existing results to only include our current selection
            processed_df = existing_processed_df[
                existing_processed_df["id"].astype(str).isin(df["id"].astype(str))
            ]
    else:
        # Process all posts
        print(f"Processing all {len(df)} selected posts")
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


# Validate API keys
if not OPENAI_API_KEY:
    print("⚠️ Warning: OpenAI API key not found in environment variables.")
    print(
        "Please set the OPENAI_API_KEY environment variable or add it to a .env file."
    )
else:
    print("✅ OpenAI API key loaded successfully.")

# Ask if we should skip existing processed data
skip_existing = (
    input("Skip processing if results already exist? (y/n, default: y): ").lower()
    != "n"
)

# Run the pilot test
run_pilot(skip_existing=skip_existing)

user_input = input("\nProceed with full pipeline run? (y/n): ")

if user_input.lower() == "y":
    run_full_pipeline(skip_existing=skip_existing)
else:
    print("Full pipeline run cancelled")

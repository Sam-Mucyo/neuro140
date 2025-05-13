#!/usr/bin/env python3
"""
Parallel LLM-Based Structurally Focused Topic Modeling

This script implements a parallel version of the LLM topic modeling pipeline,
using multiprocessing to distribute work across multiple cores and handle rate limiting.
"""

import os
import csv
import pandas as pd
import numpy as np
import re
import json
import time
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock
import queue
import random
import traceback
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel
import spacy
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import backoff
import signal
import sys

# Load environment variables from .env file
load_dotenv()

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Constants
DATA_PATH = "../data/mentalhealth_post_features_tfidf_256.csv"
RESULTS_DIR = "../results/parallel"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
RANDOM_SEED = 42  # Random seed for reproducibility
DEFAULT_MODEL = "gpt-4o-mini"  # Default model to use (supports Structured Outputs)

# Create necessary directories first
for directory in [RESULTS_DIR, CHECKPOINT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "parallel_processing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy for lemmatization
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load English stopwords
stop_words = set(stopwords.words("english"))
mental_health_stopwords = {
    "feel", "feeling", "felt", "just", "like", "know", 
    "think", "get", "got", "really",
}
stop_words = stop_words.union(mental_health_stopwords)

# Backoff handler for rate limit errors
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError),
    max_tries=8,
    factor=1.5
)
def analyze_text_with_backoff(text, model=DEFAULT_MODEL):
    """Analyze text with exponential backoff for rate limiting."""
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
            model=model,
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
        logger.error(f"Error analyzing text: {str(e)}")
        # Return a minimal result in case of error
        return {
            "themes": ["Error in analysis"],
            "emotional_tone": "unknown",
            "concerns": [],
            "cognitive_patterns": [],
            "social_context": [],
            "original_text": text,
        }

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

def create_structured_text(structured_data):
    """Create a structured text representation from the extracted data."""
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

def process_post(post_data, worker_id):
    """Process a single post."""
    try:
        post_id, text = post_data
        
        # Analyze the text
        result = analyze_text_with_backoff(text)
        
        # Add the ID to the result
        result["id"] = post_id
        
        # Create structured text representation
        structured_text = create_structured_text(result)
        result["structured_text"] = structured_text
        
        # Create tokens from structured text
        result["structured_tokens"] = lemmatize_text(structured_text)
        
        # Create tokens from original text (for baseline comparison)
        result["processed_text"] = preprocess_text(text)
        result["baseline_tokens"] = lemmatize_text(result["processed_text"])
        
        return result
    except Exception as e:
        logger.error(f"Worker {worker_id} error processing post {post_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def worker_process(worker_id, task_queue, result_dict, processed_ids, lock):
    """Worker process function that processes posts from the task queue."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore keyboard interrupts in workers
    
    worker_results = []
    checkpoint_interval = 10  # Save checkpoint after processing this many posts
    last_checkpoint_time = time.time()
    checkpoint_time_interval = 300  # Save checkpoint every 5 minutes
    
    logger.info(f"Worker {worker_id} started")
    
    try:
        while True:
            try:
                # Get a post from the queue with a timeout
                post_data = task_queue.get(timeout=1)
                
                # Check if this is a sentinel value signaling the end
                if post_data is None:
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                post_id, _ = post_data
                
                # Check if this post has already been processed
                with lock:
                    if post_id in list(processed_ids):
                        logger.info(f"Worker {worker_id} skipping already processed post {post_id}")
                        task_queue.task_done()
                        continue
                    processed_ids.append(post_id)
                
                # Process the post
                result = process_post(post_data, worker_id)
                
                if result:
                    worker_results.append(result)
                    logger.info(f"Worker {worker_id} processed post {post_id}, total: {len(worker_results)}")
                
                # Save checkpoint based on count or time interval
                current_time = time.time()
                if (len(worker_results) % checkpoint_interval == 0 or 
                    current_time - last_checkpoint_time > checkpoint_time_interval):
                    save_checkpoint(worker_results, worker_id)
                    last_checkpoint_time = current_time
                
                task_queue.task_done()
                
            except queue.Empty:
                # No more tasks in the queue, but don't exit yet as more might be added
                time.sleep(0.1)
                continue
                
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered an error: {str(e)}")
                logger.error(traceback.format_exc())
                # Save checkpoint on error
                if worker_results:
                    save_checkpoint(worker_results, worker_id)
                time.sleep(1)  # Sleep a bit before continuing
    
    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} interrupted")
    
    finally:
        # Save final results
        if worker_results:
            save_checkpoint(worker_results, worker_id, is_final=True)
        
        # Store results in the shared dictionary
        with lock:
            result_dict[worker_id] = worker_results
        
        logger.info(f"Worker {worker_id} finished, processed {len(worker_results)} posts")

def save_checkpoint(results, worker_id, is_final=False):
    """Save a checkpoint of processed results."""
    if not results:
        return
    
    prefix = "final" if is_final else "checkpoint"
    checkpoint_file = os.path.join(
        CHECKPOINT_DIR, 
        f"worker_{worker_id}_{prefix}_{len(results)}_{int(time.time())}.csv"
    )
    
    try:
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(checkpoint_file, index=False)
        logger.info(f"Worker {worker_id} saved {prefix} with {len(results)} records to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Error saving checkpoint for worker {worker_id}: {str(e)}")

def load_existing_checkpoints():
    """Load all existing checkpoint files and return combined data and processed IDs."""
    all_results = []
    processed_ids = set()
    
    # Find all checkpoint files
    checkpoint_files = [
        os.path.join(CHECKPOINT_DIR, f) 
        for f in os.listdir(CHECKPOINT_DIR) 
        if f.endswith(".csv")
    ]
    
    if not checkpoint_files:
        logger.info("No existing checkpoints found")
        return all_results, processed_ids
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load each checkpoint file
    for checkpoint_file in checkpoint_files:
        try:
            df = pd.read_csv(checkpoint_file)
            file_results = df.to_dict('records')
            all_results.extend(file_results)
            
            # Add IDs to processed set
            file_ids = set(df["id"].astype(str))
            processed_ids.update(file_ids)
            
            logger.info(f"Loaded {len(file_results)} records from {checkpoint_file}")
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_file}: {str(e)}")
    
    logger.info(f"Loaded {len(all_results)} total records from checkpoints")
    logger.info(f"Found {len(processed_ids)} unique processed IDs")
    
    return all_results, processed_ids

def merge_results(result_dict):
    """Merge results from all workers."""
    all_results = []
    
    for worker_id, results in result_dict.items():
        all_results.extend(results)
    
    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save merged results
        output_file = os.path.join(RESULTS_DIR, f"processed_data_{int(time.time())}.csv")
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(results_df)} merged records to {output_file}")
        
        return results_df
    else:
        logger.warning("No results to merge")
        return pd.DataFrame()

def run_parallel_processing(df=None, num_workers=None):
    """Run the topic modeling pipeline with parallel processing."""
    start_time = time.time()
    
    # Determine number of workers if not specified
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Cap at 16 workers to avoid API rate limits
    
    logger.info(f"Starting parallel processing with {num_workers} workers")
    
    # Load data if not provided
    if df is None:
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} records")
    
    # Ensure we have an 'id' column for tracking
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    
    # Load existing checkpoints
    existing_results, processed_ids = load_existing_checkpoints()
    
    # Create a task queue
    manager = Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()
    shared_processed_ids = manager.list(list(processed_ids))
    lock = manager.Lock()
    
    # Add tasks to the queue
    num_tasks = 0
    for idx, row in df.iterrows():
        post_id = str(row["id"]) if "id" in row else str(idx)
        if post_id not in list(shared_processed_ids):
            task_queue.put((post_id, row["post"]))
            num_tasks += 1
    
    logger.info(f"Added {num_tasks} tasks to the queue")
    
    # Add sentinel values to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, task_queue, result_dict, shared_processed_ids, lock)
        )
        p.daemon = True
        processes.append(p)
        p.start()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Merge results from all workers
        logger.info("All workers finished, merging results...")
        results_df = merge_results(result_dict)
        
        # Combine with existing results if any
        if existing_results:
            existing_df = pd.DataFrame(existing_results)
            
            # Ensure we don't have duplicate IDs
            existing_ids = set(existing_df["id"].astype(str))
            new_ids = set(results_df["id"].astype(str)) if not results_df.empty else set()
            
            # Only keep existing results that aren't in the new results
            if not new_ids.isdisjoint(existing_ids):
                existing_df = existing_df[~existing_df["id"].astype(str).isin(new_ids)]
            
            # Combine results
            if not results_df.empty:
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            else:
                combined_df = existing_df
            
            # Save combined results
            output_file = os.path.join(RESULTS_DIR, "processed_data_combined.csv")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(combined_df)} combined records to {output_file}")
            
            results_df = combined_df
        
        # Calculate and log total runtime
        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time:.2f} seconds")
        logger.info(f"Average time per post: {total_time / max(1, len(results_df)):.2f} seconds")
        
        return results_df
        
    except KeyboardInterrupt:
        logger.info("Main process interrupted")
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Still try to merge whatever results we have
        results_df = merge_results(result_dict)
        return results_df

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run parallel topic modeling")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    args = parser.parse_args()
    
    # Run the parallel processing
    results = run_parallel_processing(num_workers=args.workers)
    
    logger.info("Parallel processing completed")

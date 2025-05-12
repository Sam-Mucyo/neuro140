#!/usr/bin/env python3
"""
Token Estimator for LLM API Calls

This script estimates the number of tokens that would be used when making LLM API calls
based on a CSV file containing text data. It helps predict API costs before running
the full pipeline.

"""

import argparse
import os
import sys

import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEXT_COLUMN = "post"
DEFAULT_DATA_DIR = "data"
SYSTEM_PROMPT = """You are an AI assistant trained to analyze mental health text data.
Extract the following information from the provided text:
1. Main themes or topics
2. Emotional tone (positive, negative, neutral, mixed)
3. Key concerns or issues mentioned
4. Any cognitive patterns or distortions
5. Social context or relationships mentioned

Format your response as a structured JSON with these fields."""


def num_tokens_from_string(string: str, model_name: str = DEFAULT_MODEL) -> int:
    """
    Returns the number of tokens in a text string for a specific model.

    Args:
        string: The text string to count tokens for
        model_name: The name of the model to use for tokenization

    Returns:
        Number of tokens in the string
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(string))


def estimate_tokens_for_file(
    file_path,
    model_name=DEFAULT_MODEL,
    text_column=DEFAULT_TEXT_COLUMN,
    sample_size=None,
    random_seed=42,
):
    """
    Estimates the number of tokens that would be used for LLM API calls for a CSV file.

    Args:
        file_path: Path to the CSV file
        model_name: Name of the model to use for tokenization
        text_column: Name of the column containing the text data
        sample_size: Number of rows to sample (if None, use all data)
        random_seed: Random seed for sampling

    Returns:
        Dictionary with token statistics
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")

        # Take a sample if requested
        if sample_size is not None and sample_size < len(df):
            df = df.sample(sample_size, random_state=random_seed)
            print(f"Using a sample of {sample_size} rows")

        # Check if the text column exists
        if text_column not in df.columns:
            available_columns = ", ".join(df.columns)
            print(f"Error: Column '{text_column}' not found in the CSV file.")
            print(f"Available columns: {available_columns}")
            return None

        # Calculate tokens for each text entry
        input_tokens_per_entry = []
        estimated_output_tokens = []

        # Count tokens in the system prompt
        system_prompt_tokens = num_tokens_from_string(SYSTEM_PROMPT, model_name)

        for _, row in df.iterrows():
            # Get the text from the specified column
            text = str(row[text_column])

            # Count tokens in the text
            text_tokens = num_tokens_from_string(text, model_name)

            # Calculate total input tokens (system prompt + text)
            total_input_tokens = system_prompt_tokens + text_tokens
            input_tokens_per_entry.append(total_input_tokens)

            # Estimate output tokens (typically 20-30% of input for structured extraction)
            # Using 25% as a conservative estimate
            est_output = int(text_tokens * 0.25)
            estimated_output_tokens.append(est_output)

        # Calculate statistics
        total_input_tokens = sum(input_tokens_per_entry)
        avg_input_tokens = total_input_tokens / len(df) if len(df) > 0 else 0
        max_input_tokens = max(input_tokens_per_entry) if input_tokens_per_entry else 0
        min_input_tokens = min(input_tokens_per_entry) if input_tokens_per_entry else 0

        total_estimated_output = sum(estimated_output_tokens)

        # Calculate cost estimates using OpenAI's pricing (as of April 2025)
        # Prices are in USD per 1M tokens, converted to per 1K for calculations
        pricing = {
            "gpt-3.5-turbo": {
                "input": 3.00 / 1000,
                "output": 6.00 / 1000,
            },  # $3.00/$6.00 per 1M tokens
            "gpt-4o": {
                "input": 3.75 / 1000,
                "output": 15.00 / 1000,
            },  # $3.75/$15.00 per 1M tokens
            "gpt-4o-mini": {
                "input": 0.30 / 1000,
                "output": 1.20 / 1000,
            },  # $0.30/$1.20 per 1M tokens
            "davinci-002": {
                "input": 12.00 / 1000,
                "output": 12.00 / 1000,
            },  # $12.00/$12.00 per 1M tokens
            "babbage-002": {
                "input": 1.60 / 1000,
                "output": 1.60 / 1000,
            },  # $1.60/$1.60 per 1M tokens
        }

        model_pricing = pricing.get(model_name, pricing["gpt-3.5-turbo"])
        input_cost = (total_input_tokens / 1000) * model_pricing["input"]
        output_cost = (total_estimated_output / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "file_path": file_path,
            "num_entries": len(df),
            "total_input_tokens": total_input_tokens,
            "avg_input_tokens_per_entry": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
            "min_input_tokens": min_input_tokens,
            "total_estimated_output_tokens": total_estimated_output,
            "estimated_input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "estimated_total_cost_usd": total_cost,
            "model": model_name,
        }

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def main():
    """Main function to parse arguments and run the token estimator."""
    parser = argparse.ArgumentParser(
        description="Estimate token usage for LLM API calls based on a CSV file."
    )
    parser.add_argument("file_name", help="Name of the CSV file in the data directory")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "davinci-002",
            "babbage-002",
        ],
        help=f"Model to use for tokenization (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_TEXT_COLUMN,
        help=f"Column containing the text data (default: {DEFAULT_TEXT_COLUMN})",
    )
    parser.add_argument(
        "--sample", type=int, help="Number of rows to sample (default: use all data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the data files (default: {DEFAULT_DATA_DIR})",
    )

    args = parser.parse_args()

    # Construct the file path
    file_path = os.path.join(args.data_dir, args.file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return 1

    # Estimate tokens
    results = estimate_tokens_for_file(
        file_path=file_path,
        model_name=args.model,
        text_column=args.column,
        sample_size=args.sample,
        random_seed=args.seed,
    )

    if results:
        # Print results
        print("\n===== Token Usage Estimate =====")
        print(f"File: {results['file_path']}")
        print(f"Model: {results['model']}")
        print(f"Number of entries: {results['num_entries']}")
        print(f"Total input tokens: {results['total_input_tokens']:,}")
        print(
            f"Average input tokens per entry: {results['avg_input_tokens_per_entry']:.2f}"
        )
        print(f"Max input tokens for a single entry: {results['max_input_tokens']:,}")
        print(f"Min input tokens for a single entry: {results['min_input_tokens']:,}")
        print(
            f"Total estimated output tokens: {results['total_estimated_output_tokens']:,}"
        )
        print("\n===== Cost Estimate =====")
        print(f"Estimated input cost: ${results['estimated_input_cost_usd']:.4f}")
        print(f"Estimated output cost: ${results['estimated_output_cost_usd']:.4f}")
        print(f"Estimated total cost: ${results['estimated_total_cost_usd']:.4f}")

        # Print warning if any entries exceed token limits
        token_limits = {
            "gpt-3.5-turbo": 16385,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "davinci-002": 16385,
            "babbage-002": 16385,
        }

        limit = token_limits.get(args.model, 16385)
        if results["max_input_tokens"] > limit:
            print(
                f"\nWARNING: Some entries exceed the token limit ({limit}) for {args.model}!"
            )
            print("These entries will need to be truncated or processed differently.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

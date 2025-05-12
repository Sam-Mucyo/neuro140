#!/usr/bin/env python3
"""
Mental Health Text Analysis Pipeline

This script processes mental health-related text data using NLP techniques and API services.
It includes a pilot test section to validate the pipeline on a small subset before processing the entire dataset.
"""

import os
import csv
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Add any other API keys as needed
# OTHER_API_KEY = os.getenv("OTHER_API_KEY")

# Configuration
DATA_PATH = "data/mentalhealth_post_features_tfidf_256.csv"
RESULTS_DIR = "results"
PILOT_SAMPLE_SIZE = 5

class MentalHealthPipeline:
    """Pipeline for processing mental health text data."""
    
    def __init__(self, data_path: str = DATA_PATH):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data_path = data_path
        self.results = []
        
        # Create results directory if it doesn't exist
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        # Validate API keys
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate that required API keys are available."""
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found in environment variables.")
            # You can decide whether to raise an exception or continue with limited functionality
        
        # Add validation for other API keys as needed
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the loaded data
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            df: DataFrame containing the data to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data")
        # Add your preprocessing steps here
        # Example: df = df.dropna(subset=['text'])
        return df
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text entry.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # This is a placeholder for the actual analysis
        # Replace with your actual analysis code
        
        result = {
            "sentiment": None,
            "topics": [],
            "mental_health_indicators": [],
            "risk_assessment": None,
            # Add other analysis results as needed
        }
        
        # Example of how you might use an API
        if OPENAI_API_KEY:
            # This is just an example - replace with actual API calls
            # result["sentiment"] = call_openai_api(text)
            pass
        
        return result
    
    def process_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a batch of data.
        
        Args:
            df: DataFrame containing the batch to process
            
        Returns:
            List of results for each record
        """
        results = []
        
        for idx, row in df.iterrows():
            # Extract text from row - adjust column name as needed
            # Assuming there's a 'text' column - modify as needed
            text = row.get('text', '')
            
            # Skip empty text
            if not text:
                continue
                
            # Process the text
            result = self.analyze_text(text)
            
            # Add metadata
            result['id'] = row.get('id', idx)
            
            # Add to results
            results.append(result)
            
            # Log progress periodically
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} records")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "results.csv"):
        """
        Save results to a CSV file.
        
        Args:
            results: List of result dictionaries
            filename: Name of the output file
        """
        if not results:
            logger.warning("No results to save")
            return
            
        output_path = os.path.join(RESULTS_DIR, filename)
        logger.info(f"Saving results to {output_path}")
        
        # Get all possible keys from all dictionaries
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_keys))
            writer.writeheader()
            writer.writerows(results)
            
        logger.info(f"Saved {len(results)} records to {output_path}")
    
    def run_pilot(self) -> List[Dict[str, Any]]:
        """
        Run a pilot test on a small sample of the data.
        
        Returns:
            Results of the pilot test
        """
        logger.info(f"Running pilot test on {PILOT_SAMPLE_SIZE} records")
        
        # Load data
        df = self.load_data()
        
        # Take a small sample
        sample_df = df.head(PILOT_SAMPLE_SIZE)
        
        # Preprocess
        sample_df = self.preprocess_data(sample_df)
        
        # Process the sample
        results = self.process_batch(sample_df)
        
        # Save pilot results
        self.save_results(results, "pilot_results.csv")
        
        # Print sample of results for inspection
        logger.info("Pilot test results sample:")
        for i, result in enumerate(results[:3]):
            logger.info(f"Result {i+1}: {result}")
        
        return results
    
    def run_full_pipeline(self) -> List[Dict[str, Any]]:
        """
        Run the full pipeline on all data.
        
        Returns:
            Results from processing all data
        """
        logger.info("Running full pipeline")
        
        # Load data
        df = self.load_data()
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Process all data
        results = self.process_batch(df)
        
        # Save results
        self.save_results(results)
        
        return results


def main():
    """Main entry point for the pipeline."""
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = MentalHealthPipeline()
    
    # Run pilot test first
    logger.info("Starting pilot test")
    pilot_results = pipeline.run_pilot()
    logger.info(f"Pilot test completed with {len(pilot_results)} results")
    
    # Ask for confirmation before running the full pipeline
    user_input = input("Proceed with full pipeline run? (y/n): ")
    
    if user_input.lower() == 'y':
        # Run the full pipeline
        logger.info("Starting full pipeline")
        results = pipeline.run_full_pipeline()
        logger.info(f"Full pipeline completed with {len(results)} results")
    else:
        logger.info("Full pipeline run cancelled")
    
    # Calculate and log total runtime
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

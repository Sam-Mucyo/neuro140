# Source Code Overview

This directory contains the Python scripts that implement the LLM-based structurally focused topic modeling pipeline:

## llm_structured_topic_modeling.py

- Serial pipeline:
  - Data loading & preprocessing (CSV → pandas + spaCy/NLTK)
  - LLM-driven structured extraction (OpenAI Structured Outputs)
  - Batch processing with checkpoints to avoid re-calling the LLM
  - LDA modeling & evaluation (Gensim)
  - Orchestration: pilot demo, baseline vs structured LDA, result saving

**Why Parallelize the above?** After running the above on sample documents, we noticed that processing one posts took around 4-5 seconds. Worst case scenario, the serial pipeline processing all ∼13k posts would take over 20 hours, risking long runtimes and loss of progress on interruptions. The parallel pipeline distributes LLM calls across multiple processes, uses exponential backoff for API rate limits, and checkpoints intermediate results for robust, resumable execution.

## parallel_topic_modeling.py

- Parallel pipeline:
  - Multiprocessing + task queue for concurrent LLM calls
  - Exponential backoff for API rate limits
  - Worker checkpoints & resume
  - Result merging & deduplication with graceful shutdown handling

## merge_results.py

- Merges partial checkpoint CSVs into one CSV
- Deduplicates by post ID
- Generates basic statistics (theme counts, emotional tone distributions)

## estimate_tokens.py

- Estimates token usage & API call cost using tiktoken
- Calculates input/output token counts based on a system prompt and post text
- CLI interface for model selection, sampling, and cost breakdown

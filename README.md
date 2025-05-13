<!--- Root-level README for the entire repository --->

# LLM-Based Topic Modeling and Analysis

This repository implements and compares traditional LDA topic modeling with an LLM-structured approach on mental health–related Reddit data.

## Repository Structure

```plaintext
- .github/                # GitHub workflows
- project/                # Core code, data, and experiments
  - experiments/          # Jupyter notebooks for EDA and modeling (see `project/experiments/README.md`)
  - src/                  # Python scripts for serial & parallel pipelines, utilities (see `project/src/README.md`)
```

## Getting Started

1. Clone this repository:
   ```bash
   git clone
   ```
2. Follow
3. Follow the environment setup instructions in `project/README.md` to create a virtual environment or Conda environment.
4. Explore the analyses:
   - Notebooks: `project/experiments/`
   - Serial pipeline: `python project/src/llm_structured_topic_modeling.py`
   - Parallel pipeline: `python project/src/parallel_topic_modeling.py`
   - Merge results: `python project/src/merge_results.py`

## Further Reading

- See `project/README.md` for detailed instructions on the LLM-based topic modeling pipeline.
- See `project/experiments/README.md` for notebook walkthroughs.

## Contributing

Contributions and issues are welcome!

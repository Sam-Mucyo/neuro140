# Experiment Notebooks Overview

This directory contains Jupyter notebooks for exploring and comparing topic modeling workflows:

1. `eda.ipynb`

   - Exploratory Data Analysis on raw Reddit mental health posts
   - Post-length distributions, word clouds, sentiment/readability stats, filtering decisions

2. `traditional_lda.ipynb`

   - Baseline LDA pipeline on raw text
   - Tokenization, lemmatization, dictionary & corpus creation (spaCy + Gensim)
   - LDA training (varying number of topics), coherence & perplexity evaluation
   - Topic inspection and visualization

3. `comparative_analysis.ipynb`
   - Compares Baseline vs Structured LDA results
   - Plots bar charts for coherence, diversity, perplexity
   - Embeds pyLDAvis visualizations and word clouds
   - Side-by-side representative documents for qualitative assessment

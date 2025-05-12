#!/usr/bin/env python3
"""
Topic Model Visualization Utilities

This module provides functions for visualizing topic models:
- Topic word clouds
- Topic similarity heatmaps
- Document-topic distribution visualizations
- Comparative visualizations between models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Any, Optional
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


def generate_topic_wordcloud(
    model,
    topic_id,
    dictionary=None,
    background_color="white",
    width=800,
    height=400,
    max_words=100,
):
    """
    Generate a word cloud for a specific topic.

    Args:
        model: Gensim LDA model
        topic_id: ID of the topic to visualize
        dictionary: Gensim dictionary (optional)
        background_color: Background color for the word cloud
        width: Width of the word cloud image
        height: Height of the word cloud image
        max_words: Maximum number of words to include

    Returns:
        WordCloud object
    """
    # Get topic word distribution
    topic_words = dict(model.show_topic(topic_id, max_words))

    # Generate word cloud
    wordcloud = WordCloud(
        background_color=background_color,
        width=width,
        height=height,
        max_words=max_words,
        colormap="viridis",
        prefer_horizontal=1.0,
    ).generate_from_frequencies(topic_words)

    return wordcloud


def plot_topic_wordclouds(model, num_topics=None, cols=2, figsize=(15, 20)):
    """
    Plot word clouds for multiple topics in a grid.

    Args:
        model: Gensim LDA model
        num_topics: Number of topics to plot (default: all)
        cols: Number of columns in the grid
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    if num_topics is None:
        num_topics = model.num_topics

    rows = int(np.ceil(num_topics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_topics):
        if i < len(axes):
            wordcloud = generate_topic_wordcloud(model, i)
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(f"Topic {i+1}")
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(num_topics, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_topic_similarity_heatmap(model, figsize=(10, 8)):
    """
    Plot a heatmap showing similarity between topics.

    Args:
        model: Gensim LDA model
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    num_topics = model.num_topics
    topic_vectors = [model.get_topic_terms(i, 100) for i in range(num_topics)]

    # Convert to term-weight dictionaries
    topic_dicts = []
    for vec in topic_vectors:
        topic_dict = {term_id: weight for term_id, weight in vec}
        topic_dicts.append(topic_dict)

    # Calculate cosine similarity matrix
    similarity_matrix = np.zeros((num_topics, num_topics))

    for i in range(num_topics):
        for j in range(num_topics):
            # Get common terms
            common_terms = set(topic_dicts[i].keys()) & set(topic_dicts[j].keys())

            # Calculate dot product
            dot_product = sum(
                topic_dicts[i][term] * topic_dicts[j][term] for term in common_terms
            )

            # Calculate magnitudes
            mag_i = np.sqrt(sum(w**2 for w in topic_dicts[i].values()))
            mag_j = np.sqrt(sum(w**2 for w in topic_dicts[j].values()))

            # Calculate cosine similarity
            if mag_i > 0 and mag_j > 0:
                similarity_matrix[i, j] = dot_product / (mag_i * mag_j)
            else:
                similarity_matrix[i, j] = 0

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="YlGnBu",
        xticklabels=[f"Topic {i+1}" for i in range(num_topics)],
        yticklabels=[f"Topic {i+1}" for i in range(num_topics)],
        ax=ax,
    )
    ax.set_title("Topic Similarity Matrix")

    return fig


def plot_document_topic_distribution(
    model, corpus, doc_indices=None, top_n=5, figsize=(12, 8)
):
    """
    Plot topic distribution for selected documents.

    Args:
        model: Gensim LDA model
        corpus: Gensim corpus
        doc_indices: Indices of documents to plot (default: first 5)
        top_n: Number of top topics to display per document
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    if doc_indices is None:
        doc_indices = list(range(min(5, len(corpus))))

    num_docs = len(doc_indices)
    fig, axes = plt.subplots(num_docs, 1, figsize=figsize)

    if num_docs == 1:
        axes = [axes]

    for i, doc_idx in enumerate(doc_indices):
        if doc_idx < len(corpus):
            # Get document-topic distribution
            doc_topics = model.get_document_topics(corpus[doc_idx])
            doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)[:top_n]

            # Extract topics and probabilities
            topics = [f"Topic {t+1}" for t, _ in doc_topics]
            probs = [p for _, p in doc_topics]

            # Plot bar chart
            axes[i].barh(topics, probs, color=plt.cm.tab10(np.arange(len(topics)) % 10))
            axes[i].set_title(f"Document {doc_idx} - Topic Distribution")
            axes[i].set_xlabel("Probability")
            axes[i].set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_topic_prevalence(model, corpus, figsize=(12, 8)):
    """
    Plot the prevalence of each topic across the corpus.

    Args:
        model: Gensim LDA model
        corpus: Gensim corpus
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    # Calculate topic prevalence
    topic_counts = np.zeros(model.num_topics)

    for doc in corpus:
        doc_topics = model.get_document_topics(doc)
        for topic_id, prob in doc_topics:
            topic_counts[topic_id] += prob

    # Normalize
    topic_prevalence = topic_counts / len(corpus)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        [f"Topic {i+1}" for i in range(model.num_topics)],
        topic_prevalence,
        color=plt.cm.viridis(np.linspace(0, 1, model.num_topics)),
    )
    ax.set_title("Topic Prevalence Across Corpus")
    ax.set_ylabel("Average Topic Probability")
    ax.set_xlabel("Topics")
    plt.xticks(rotation=45)

    return fig


def compare_topic_prevalence(models_data, figsize=(15, 8)):
    """
    Compare topic prevalence across different models.

    Args:
        models_data: List of dictionaries with keys:
            - name: Model name
            - model: Gensim LDA model
            - corpus: Gensim corpus
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(len(models_data), 1, figsize=figsize)

    if len(models_data) == 1:
        axes = [axes]

    for i, model_info in enumerate(models_data):
        model = model_info["model"]
        corpus = model_info["corpus"]
        name = model_info["name"]

        # Calculate topic prevalence
        topic_counts = np.zeros(model.num_topics)

        for doc in corpus:
            doc_topics = model.get_document_topics(doc)
            for topic_id, prob in doc_topics:
                topic_counts[topic_id] += prob

        # Normalize
        topic_prevalence = topic_counts / len(corpus)

        # Plot
        axes[i].bar(
            [f"Topic {i+1}" for i in range(model.num_topics)],
            topic_prevalence,
            color=plt.cm.viridis(np.linspace(0, 1, model.num_topics)),
        )
        axes[i].set_title(f"{name} - Topic Prevalence")
        axes[i].set_ylabel("Average Topic Probability")
        axes[i].set_xlabel("Topics")
        plt.sca(axes[i])
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

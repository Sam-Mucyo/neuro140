#!/usr/bin/env python3
"""
Topic Model Evaluation Utilities

This module provides functions for evaluating topic models using various metrics:
- Coherence scores (c_v and u_mass)
- Topic diversity
- Perplexity
- Topic quality visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import CoherenceModel
from typing import List, Dict, Tuple, Any, Optional
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


def calculate_coherence_cv(model, texts, dictionary):
    """
    Calculate the c_v coherence score for a topic model.
    
    Args:
        model: Gensim LDA model
        texts: List of tokenized texts
        dictionary: Gensim dictionary
        
    Returns:
        float: c_v coherence score
    """
    coherence_model = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    return coherence_model.get_coherence()


def calculate_coherence_umass(model, corpus, dictionary):
    """
    Calculate the u_mass coherence score for a topic model.
    
    Args:
        model: Gensim LDA model
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        
    Returns:
        float: u_mass coherence score
    """
    coherence_model = CoherenceModel(
        model=model, 
        corpus=corpus, 
        dictionary=dictionary, 
        coherence='u_mass'
    )
    return coherence_model.get_coherence()


def calculate_topic_diversity(model, num_words=10):
    """
    Calculate the topic diversity as the ratio of unique words 
    in top N words across all topics.
    
    Args:
        model: Gensim LDA model
        num_words: Number of top words to consider per topic
        
    Returns:
        float: Topic diversity score (0-1)
    """
    topics = [dict(model.show_topic(topicid, num_words)) for topicid in range(model.num_topics)]
    unique_words = set()
    for topic in topics:
        unique_words.update(topic.keys())
    
    diversity = len(unique_words) / (len(topics) * num_words)
    return diversity


def create_corpus_from_tokens(tokens_list, dictionary):
    """
    Create a gensim corpus from a list of tokens.
    
    Args:
        tokens_list: List of token lists
        dictionary: Gensim dictionary
        
    Returns:
        list: Gensim corpus
    """
    return [dictionary.doc2bow(tokens) for tokens in tokens_list]


def prepare_pyldavis_visualization(model, corpus, dictionary):
    """
    Prepare pyLDAvis visualization data.
    
    Args:
        model: Gensim LDA model
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        
    Returns:
        pyLDAvis visualization data
    """
    return gensimvis.prepare(model, corpus, dictionary)


def compare_topic_models(models_data, metrics=None):
    """
    Compare multiple topic models using specified metrics.
    
    Args:
        models_data: Dictionary of model data with keys:
            - name: Model name
            - model: Gensim LDA model
            - corpus: Gensim corpus
            - dictionary: Gensim dictionary
            - texts: List of tokenized texts
        metrics: List of metrics to compare (default: all)
        
    Returns:
        DataFrame: Comparison results
    """
    if metrics is None:
        metrics = ['c_v', 'u_mass', 'diversity', 'perplexity']
    
    results = []
    
    for model_info in models_data:
        model_results = {'Model': model_info['name']}
        
        if 'c_v' in metrics:
            model_results['Coherence (c_v)'] = calculate_coherence_cv(
                model_info['model'], 
                model_info['texts'], 
                model_info['dictionary']
            )
        
        if 'u_mass' in metrics:
            model_results['Coherence (u_mass)'] = calculate_coherence_umass(
                model_info['model'], 
                model_info['corpus'], 
                model_info['dictionary']
            )
        
        if 'diversity' in metrics:
            model_results['Topic Diversity'] = calculate_topic_diversity(model_info['model'])
        
        if 'perplexity' in metrics:
            model_results['Perplexity'] = model_info['model'].log_perplexity(model_info['corpus'])
        
        results.append(model_results)
    
    return pd.DataFrame(results)


def plot_topic_word_distributions(models, top_n=10, figsize=(15, 10)):
    """
    Plot the top words for each topic across different models.
    
    Args:
        models: Dictionary mapping model names to Gensim LDA models
        top_n: Number of top words to display per topic
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(len(models), 1, figsize=figsize)
    
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i] if len(models) > 1 else axes
        
        # Get top words for each topic
        topic_words = []
        for t in range(model.num_topics):
            top_words = [word for word, _ in model.show_topic(t, top_n)]
            topic_words.append(' '.join(top_words))
        
        # Plot as horizontal bar
        y_pos = np.arange(len(topic_words))
        ax.barh(y_pos, [1] * len(topic_words), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Topic {i+1}' for i in range(len(topic_words))])
        ax.invert_yaxis()
        ax.set_xlabel('Words')
        ax.set_title(f'{model_name} - Top {top_n} Words per Topic')
        
        # Add text labels
        for i, words in enumerate(topic_words):
            ax.text(0.01, i, words, va='center')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(comparison_df, figsize=(12, 8)):
    """
    Plot comparison metrics for different models.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    metrics = [col for col in comparison_df.columns if col != 'Model']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    for i, metric in enumerate(metrics):
        ax = axes[i] if n_metrics > 1 else axes
        sns.barplot(x='Model', y=metric, data=comparison_df, ax=ax)
        ax.set_title(f'Comparison of {metric}')
        ax.set_ylabel(metric)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def extract_topic_keywords(model, num_topics=None, num_words=10):
    """
    Extract top keywords for each topic in a model.
    
    Args:
        model: Gensim LDA model
        num_topics: Number of topics to extract (default: all)
        num_words: Number of words per topic
        
    Returns:
        dict: Mapping of topic IDs to lists of top words
    """
    if num_topics is None:
        num_topics = model.num_topics
    
    topics = {}
    for i in range(num_topics):
        topic_words = [word for word, _ in model.show_topic(i, num_words)]
        topics[i] = topic_words
    
    return topics

"""
Similarity metrics for text comparison
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts using TF-IDF vectorization.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Transform texts into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def calculate_jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two texts (word-level).
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    if not words1 and not words2:
        return 1.0  # If both are empty, they're identical
    
    try:
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union
    except Exception as e:
        print(f"Error calculating Jaccard similarity: {e}")
        return 0.0

def calculate_semantic_similarity(text1, text2):
    """
    Calculate pseudo-semantic similarity by comparing word overlap patterns.
    
    This is a simplified approach that doesn't use embedding models like Word2Vec or BERT.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Semantic similarity score between 0 and 1
    """
    # For now, this is a weighted combination of cosine and Jaccard similarity
    # In a real app, you'd use a proper semantic model
    cosine = calculate_cosine_similarity(text1, text2)
    jaccard = calculate_jaccard_similarity(text1, text2)
    
    # Weight more towards cosine similarity
    return 0.7 * cosine + 0.3 * jaccard

def calculate_similarity(text1, text2, metrics=None):
    """
    Calculate various similarity metrics between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        metrics (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary of similarity scores
    """
    if metrics is None:
        metrics = ["Cosine Similarity", "Jaccard Similarity", "Semantic Similarity"]
    
    results = {}
    
    if "Cosine Similarity" in metrics:
        results["cosine_similarity"] = calculate_cosine_similarity(text1, text2)
    
    if "Jaccard Similarity" in metrics:
        results["jaccard_similarity"] = calculate_jaccard_similarity(text1, text2)
    
    if "Semantic Similarity" in metrics:
        results["semantic_similarity"] = calculate_semantic_similarity(text1, text2)
    
    return results

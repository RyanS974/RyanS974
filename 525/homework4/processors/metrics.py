import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess text for similarity calculations
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_cosine_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Cosine similarity score
    """
    # Preprocess texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return float(cosine_sim)

def calculate_jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Jaccard similarity score
    """
    # Preprocess texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)
    
    # Tokenize
    tokens1 = set(word_tokenize(preprocessed_text1))
    tokens2 = set(word_tokenize(preprocessed_text2))
    
    # Calculate Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def calculate_ngram_overlap(text1, text2, n=2):
    """
    Calculate n-gram overlap between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        n (int): Size of n-grams
        
    Returns:
        float: N-gram overlap score
    """
    # Preprocess texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)
    
    # Tokenize
    tokens1 = word_tokenize(preprocessed_text1)
    tokens2 = word_tokenize(preprocessed_text2)
    
    # Generate n-grams
    ngrams1 = set(' '.join(gram) for gram in ngrams(tokens1, n))
    ngrams2 = set(' '.join(gram) for gram in ngrams(tokens2, n))
    
    # Calculate overlap
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts
    
    Note: In a real implementation, this would use a pretrained language model.
    This is a simplified version for demonstration purposes.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Semantic similarity score
    """
    # For demonstration, use a weighted combination of other similarities
    cosine_sim = calculate_cosine_similarity(text1, text2)
    jaccard_sim = calculate_jaccard_similarity(text1, text2)
    ngram_sim = calculate_ngram_overlap(text1, text2, 2)
    
    # Weighted average (would be replaced with actual embedding-based similarity)
    semantic_sim = (0.5 * cosine_sim) + (0.3 * jaccard_sim) + (0.2 * ngram_sim)
    
    return float(semantic_sim)

def calculate_lexical_diversity(text):
    """
    Calculate lexical diversity (type-token ratio)
    
    Args:
        text (str): Input text
        
    Returns:
        float: Lexical diversity score
    """
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize
    tokens = word_tokenize(preprocessed_text)
    
    # Calculate type-token ratio
    if len(tokens) == 0:
        return 0.0
    
    return len(set(tokens)) / len(tokens)

def calculate_complexity(text):
    """
    Calculate linguistic complexity metrics
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Complexity metrics
    """
    # Preprocess minimally to keep sentence structure
    text_lower = text.lower()
    
    # Tokenize into sentences and words
    sentences = nltk.sent_tokenize(text_lower)
    words = word_tokenize(text_lower)
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    
    # Calculate lexical diversity
    lexical_diversity = calculate_lexical_diversity(text)
    
    return {
        "avg_sentence_length": float(avg_sentence_length),
        "avg_word_length": float(avg_word_length),
        "lexical_diversity": float(lexical_diversity)
    }

def calculate_similarity(text1, text2, methods=None):
    """
    Calculate similarity between texts using various methods
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        methods (list): List of similarity methods to apply
        
    Returns:
        dict: Similarity metrics
    """
    if methods is None:
        methods = ["Cosine Similarity"]
    
    results = {}
    
    if "Cosine Similarity" in methods:
        results["cosine_similarity"] = calculate_cosine_similarity(text1, text2)
    
    if "Jaccard Similarity" in methods:
        results["jaccard_similarity"] = calculate_jaccard_similarity(text1, text2)
    
    if "N-gram Overlap" in methods:
        for n in range(1, 4):
            results[f"{n}-gram_overlap"] = calculate_ngram_overlap(text1, text2, n)
    
    if "Semantic Similarity" in methods:
        results["semantic_similarity"] = calculate_semantic_similarity(text1, text2)
    
    # Add complexity comparison
    if "Complexity Comparison" in methods:
        complexity1 = calculate_complexity(text1)
        complexity2 = calculate_complexity(text2)
        
        results["complexity_comparison"] = {
            "text1_complexity": complexity1,
            "text2_complexity": complexity2,
            "complexity_difference": {
                "avg_sentence_length": complexity1["avg_sentence_length"] - complexity2["avg_sentence_length"],
                "avg_word_length": complexity1["avg_word_length"] - complexity2["avg_word_length"],
                "lexical_diversity": complexity1["lexical_diversity"] - complexity2["lexical_diversity"]
            }
        }
    
    return results

def calculate_diversity(text):
    """
    Calculate lexical diversity and other metrics
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Diversity metrics
    """
    return calculate_complexity(text)
    vector
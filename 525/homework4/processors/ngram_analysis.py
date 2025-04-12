from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text for n-gram analysis
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (but keep spaces and punctuation for n-grams)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords for unigrams, but keep for n-grams (important for context)
    # stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def extract_ngrams(text, n=2):
    """
    Extract n-grams from text
    
    Args:
        text (str): Input text
        n (int): Size of n-grams to extract
        
    Returns:
        dict: N-grams with counts
    """
    # Preprocess text
    tokens = preprocess_text(text)
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Convert n-grams to strings for easier handling
    n_gram_strings = [' '.join(gram) for gram in n_grams]
    
    # Count occurrences
    gram_counts = Counter(n_gram_strings)
    
    return dict(gram_counts)

def compare_ngrams(texts, model_names, n=2, top_n=10):
    """
    Compare n-grams across different texts
    
    Args:
        texts (list): List of text responses
        model_names (list): Names of models corresponding to responses
        n (int): Size of n-grams to extract
        top_n (int): Number of top n-grams to consider
        
    Returns:
        dict: Comparative analysis
    """
    # Extract n-grams for each text
    model_ngrams = {}
    for i, (text, model) in enumerate(zip(texts, model_names)):
        model_ngrams[model] = extract_ngrams(text, n)
    
    # Get top n-grams for each model
    top_ngrams = {}
    for model, ngrams_dict in model_ngrams.items():
        sorted_ngrams = sorted(ngrams_dict.items(), key=lambda x: x[1], reverse=True)
        top_ngrams[model] = [{"ngram": ngram, "count": count} for ngram, count in sorted_ngrams[:top_n]]
    
    # Find unique n-grams for each model
    unique_ngrams = {}
    for i, model1 in enumerate(model_names):
        # Get all n-grams from other models
        other_ngrams = set()
        for j, model2 in enumerate(model_names):
            if i != j:
                other_ngrams.update(model_ngrams[model2].keys())
        
        # Find n-grams unique to this model
        unique_to_model = set(model_ngrams[model1].keys()) - other_ngrams
        
        # Sort by count
        sorted_unique = sorted(
            [(ngram, model_ngrams[model1][ngram]) for ngram in unique_to_model],
            key=lambda x: x[1],
            reverse=True
        )
        
        unique_ngrams[model1] = [{"ngram": ngram, "count": count} for ngram, count in sorted_unique[:top_n]]
    
    # Calculate pairwise similarity between models
    similarities = {}
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if j <= i:  # Avoid duplicate comparisons
                continue
            
            # Get sets of n-grams
            ngrams1 = set(model_ngrams[model1].keys())
            ngrams2 = set(model_ngrams[model2].keys())
            
            # Calculate Jaccard similarity
            intersection = ngrams1.intersection(ngrams2)
            union = ngrams1.union(ngrams2)
            
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0
            
            similarities[f"{model1} vs {model2}"] = {
                "jaccard_similarity": jaccard,
                "common_ngrams": list(intersection)[:top_n]
            }
    
    # Create n-gram frequency matrix for comparison
    all_ngrams = set()
    for model_dict in model_ngrams.values():
        all_ngrams.update(model_dict.keys())
    
    # Calculate ngram variances to find most differential ngrams
    ngram_variances = {}
    for ngram in all_ngrams:
        counts = [model_dict.get(ngram, 0) for model_dict in model_ngrams.values()]
        if len(counts) > 1:
            ngram_variances[ngram] = np.var(counts)
    
    # Get top differential ngrams
    top_diff_ngrams = sorted(ngram_variances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    differential_ngrams = [ngram for ngram, _ in top_diff_ngrams]
    
    # Create matrix of counts for top differential ngrams
    ngram_matrix = {}
    for ngram in differential_ngrams:
        ngram_matrix[ngram] = {model: model_dict.get(ngram, 0) for model, model_dict in model_ngrams.items()}
    
    # Format results
    result = {
        "n": n,
        "top_ngrams": top_ngrams,
        "unique_ngrams": unique_ngrams,
        "similarities": similarities,
        "differential_ngrams": differential_ngrams,
        "ngram_matrix": ngram_matrix,
        "models": model_names
    }
    
    return result

def unique_ngrams(text1, text2, n=2):
    """
    Find unique n-grams in one text vs another
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        n (int): Size of n-grams
        
    Returns:
        dict: N-grams unique to each text
    """
    # Extract n-grams
    ngrams1 = extract_ngrams(text1, n)
    ngrams2 = extract_ngrams(text2, n)
    
    # Find unique n-grams
    unique_to_1 = set(ngrams1.keys()) - set(ngrams2.keys())
    unique_to_2 = set(ngrams2.keys()) - set(ngrams1.keys())
    
    # Sort by frequency
    sorted_unique_1 = sorted(
        [(ngram, ngrams1[ngram]) for ngram in unique_to_1],
        key=lambda x: x[1],
        reverse=True
    )
    
    sorted_unique_2 = sorted(
        [(ngram, ngrams2[ngram]) for ngram in unique_to_2],
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "unique_to_first": [{"ngram": ngram, "count": count} for ngram, count in sorted_unique_1[:10]],
        "unique_to_second": [{"ngram": ngram, "count": count} for ngram, count in sorted_unique_2[:10]]
    }

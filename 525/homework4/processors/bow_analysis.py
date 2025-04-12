from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text for bag of words analysis
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Filter out short words (likely not meaningful)
    tokens = [token for token in tokens if len(token) > 2]
    
    # Join back to string
    return ' '.join(tokens)

def create_bow(text):
    """
    Create bag of words representation
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Bag of words representation with word counts
    """
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize
    tokens = preprocessed_text.split()
    
    # Count occurrences
    word_counts = Counter(tokens)
    
    return dict(word_counts)

def compare_bow(bow1, bow2):
    """
    Compare two bag of words representations
    
    Args:
        bow1 (dict): First bag of words
        bow2 (dict): Second bag of words
        
    Returns:
        dict: Comparison metrics
    """
    # Get all unique words
    all_words = set(bow1.keys()).union(set(bow2.keys()))
    
    # Words in both
    common_words = set(bow1.keys()).intersection(set(bow2.keys()))
    
    # Words unique to each
    unique_to_1 = set(bow1.keys()) - set(bow2.keys())
    unique_to_2 = set(bow2.keys()) - set(bow1.keys())
    
    # Calculate Jaccard similarity
    jaccard = len(common_words) / len(all_words) if len(all_words) > 0 else 0
    
    # Calculate cosine similarity
    vec1 = np.zeros(len(all_words))
    vec2 = np.zeros(len(all_words))
    
    for i, word in enumerate(all_words):
        vec1[i] = bow1.get(word, 0)
        vec2[i] = bow2.get(word, 0)
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        cosine = 0
    else:
        cosine = np.dot(vec1, vec2) / (norm1 * norm2)
    
    return {
        "jaccard_similarity": jaccard,
        "cosine_similarity": cosine,
        "common_word_count": len(common_words),
        "unique_to_first": list(unique_to_1)[:20],  # Limit for readability
        "unique_to_second": list(unique_to_2)[:20]  # Limit for readability
    }

def important_words(bow, top_n=10):
    """
    Extract most important/distinctive words
    
    Args:
        bow (dict): Bag of words representation
        top_n (int): Number of top words to return
        
    Returns:
        list: Top words with counts
    """
    # Sort by count
    sorted_words = sorted(bow.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N
    return [{"word": word, "count": count} for word, count in sorted_words[:top_n]]

def compare_bow_across_texts(texts, model_names, top_n=25):
    """
    Compare bag of words across multiple texts
    
    Args:
        texts (list): List of text responses
        model_names (list): List of model names corresponding to responses
        top_n (int): Number of top words to include
        
    Returns:
        dict: Comparative bag of words analysis
    """
    # Create bag of words for each text
    bows = [create_bow(text) for text in texts]
    
    # Map to models
    model_bows = {model: bow for model, bow in zip(model_names, bows)}
    
    # Get important words for each model
    model_important_words = {model: important_words(bow, top_n) for model, bow in model_bows.items()}
    
    # Compare pairwise
    comparisons = {}
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if j <= i:  # Avoid duplicate comparisons
                continue
            
            comparison_key = f"{model1} vs {model2}"
            comparisons[comparison_key] = compare_bow(model_bows[model1], model_bows[model2])
    
    # Create combined word list across all models
    all_words = set()
    for bow in bows:
        all_words.update(bow.keys())
    
    # Create a matrix of word counts across models
    word_count_matrix = {}
    for word in sorted(list(all_words)):
        word_counts = [bow.get(word, 0) for bow in bows]
        # Only include words that show up in at least one model
        if any(count > 0 for count in word_counts):
            word_count_matrix[word] = {model: bow.get(word, 0) for model, bow in zip(model_names, bows)}
    
    # Sort matrix by most differential words (words with biggest variance across models)
    word_variances = {}
    for word, counts in word_count_matrix.items():
        count_values = list(counts.values())
        if len(count_values) > 1:
            word_variances[word] = np.var(count_values)
    
    # Get top differential words
    top_diff_words = sorted(word_variances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    differential_words = [word for word, _ in top_diff_words]
    
    # Format results
    result = {
        "model_word_counts": model_bows,
        "important_words": model_important_words,
        "comparisons": comparisons,
        "differential_words": differential_words,
        "word_count_matrix": {word: word_count_matrix[word] for word in differential_words},
        "models": model_names
    }
    
    return result

def compare_bow(texts, model_names, top_n=25):
    """
    Compare bag of words between different texts
    
    Args:
        texts (list): List of text responses to compare
        model_names (list): Names of models corresponding to responses
        top_n (int): Number of top words to consider
        
    Returns:
        dict: Comparative analysis
    """
    return compare_bow_across_texts(texts, model_names, top_n)
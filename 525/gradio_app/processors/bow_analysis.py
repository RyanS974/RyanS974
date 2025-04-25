"""
Updated bow_analysis.py to include similarity metrics.
Preprocessing here is more advanced than n-gram version.
Lowercase, tokenize, remove stopwords, non-alphabetic characters removal, short words removal, lemmatization.
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from processors.metrics import calculate_similarity

# not used currently imports, but left in case I start using them again
import numpy as np
from collections import Counter
import re
import nltk

# Define the compare_bow_across_texts function directly in this file
def compare_bow_across_texts(texts, model_names, top_n=25):
    """
    Compare bag of words representations across multiple texts.
    
    Args:
        texts (list): List of text responses to compare
        model_names (list): Names of models corresponding to responses
        top_n (int): Number of top words to consider
        
    Returns:
        dict: Bag of words analysis results
    """
    # Initialize the results dictionary
    result = {
        "models": model_names,
        "important_words": {},
        "word_count_matrix": {},
        "differential_words": []
    }
    
    # Make sure we have texts to analyze
    if not texts or len(texts) < 1:
        return result
        
    # Preprocess texts (tokenize, remove stopwords, etc.)
    preprocessed_texts = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    for text in texts:
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords, punctuation, and lemmatize
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words and len(token) > 2:
                filtered_tokens.append(lemmatizer.lemmatize(token))
        
        preprocessed_texts.append(" ".join(filtered_tokens))
    
    # Create bag of words representations using CountVectorizer
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(preprocessed_texts)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create word count matrix
    word_counts = {}
    for i, model in enumerate(model_names):
        counts = X[i].toarray()[0]
        word_counts[model] = {}
        
        # Store word frequencies for this model
        for j, word in enumerate(feature_names):
            if counts[j] > 0:  # Only store words that appear
                word_counts[model][word] = int(counts[j])
                
                # Add to word count matrix
                if word not in result["word_count_matrix"]:
                    result["word_count_matrix"][word] = {}
                result["word_count_matrix"][word][model] = int(counts[j])
    
    # Find important words for each model
    for model, word_freq in word_counts.items():
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Store top N words
        result["important_words"][model] = [
            {"word": word, "count": count} 
            for word, count in sorted_words[:top_n]
        ]
    
    # Calculate differential words (words with biggest frequency difference between models)
    if len(model_names) >= 2:
        model1, model2 = model_names[0], model_names[1]
        
        # Calculate differences
        diff_scores = {}
        for word in result["word_count_matrix"]:
            count1 = result["word_count_matrix"][word].get(model1, 0)
            count2 = result["word_count_matrix"][word].get(model2, 0)
            
            # Absolute difference
            diff_scores[word] = abs(count1 - count2)
        
        # Sort by difference
        sorted_diffs = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)
        result["differential_words"] = [word for word, _ in sorted_diffs[:top_n]]
        
        # Calculate overlap statistics
        model1_words = set(word_counts.get(model1, {}).keys())
        model2_words = set(word_counts.get(model2, {}).keys())
        common_words = model1_words.intersection(model2_words)
        
        # Initialize comparisons if needed
        if "comparisons" not in result:
            result["comparisons"] = {}
            
        comparison_key = f"{model1} vs {model2}"
        result["comparisons"][comparison_key] = {
            "common_word_count": len(common_words)
        }
    
    return result

def add_similarity_metrics(bow_results, response_texts, model_names):
    """
    Add similarity metrics to the bag of words analysis results
    
    Args:
        bow_results (dict): The bag of words analysis results
        response_texts (list): List of response texts to compare
        model_names (list): List of model names corresponding to responses
        
    Returns:
        dict: Updated bag of words results with similarity metrics
    """
    # Make sure we have at least two responses to compare
    if len(response_texts) < 2 or len(model_names) < 2:
        print("Need at least two responses to calculate similarity metrics")
        return bow_results
    
    # Get the first two responses (current implementation only handles two-way comparisons)
    text1, text2 = response_texts[0], response_texts[1]
    model1, model2 = model_names[0], model_names[1]
    
    # Generate the comparison key
    comparison_key = f"{model1} vs {model2}"
    
    # Initialize comparisons if needed
    if "comparisons" not in bow_results:
        bow_results["comparisons"] = {}
    
    # Initialize the comparison entry if needed
    if comparison_key not in bow_results["comparisons"]:
        bow_results["comparisons"][comparison_key] = {}
    
    # Calculate similarity metrics
    metrics = calculate_similarity(text1, text2)
    
    # Add metrics to the comparison
    bow_results["comparisons"][comparison_key].update({
        "cosine_similarity": metrics.get("cosine_similarity", 0),
        "jaccard_similarity": metrics.get("jaccard_similarity", 0),
        "semantic_similarity": metrics.get("semantic_similarity", 0)
    })
    
    # If we have common_word_count from BOW analysis, keep it
    if "common_word_count" not in bow_results["comparisons"][comparison_key]:
        # Calculate from bow data as a fallback
        if "important_words" in bow_results:
            words1 = set([item["word"] for item in bow_results["important_words"].get(model1, [])])
            words2 = set([item["word"] for item in bow_results["important_words"].get(model2, [])])
            common_words = words1.intersection(words2)
            bow_results["comparisons"][comparison_key]["common_word_count"] = len(common_words)
    
    return bow_results

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
    bow_results = compare_bow_across_texts(texts, model_names, top_n)
    
    # Add similarity metrics to the results
    if len(texts) >= 2 and len(model_names) >= 2:
        bow_results = add_similarity_metrics(bow_results, texts, model_names)
    
    return bow_results
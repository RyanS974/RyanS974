"""
N-gram analysis for comparing text responses.
Minimal preprocessing is done here, basically just removing stop words and tokenization. From my research this is a good combination for n-gram analysis.
"""
from sklearn.feature_extraction.text import CountVectorizer

# these aren't used currently, as they were imports for testing versions with them. the code is removed also, but I decided to just leave these imports incase I start using them again.
from collections import Counter
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Helper function to flatten nested lists
def flatten_list(nested_list):
    """
    Recursively flattens a nested list.

    Args:
        nested_list (list): A potentially nested list.

    Returns:
        list: A flattened list.
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item

def compare_ngrams(texts, model_names, n=2, top_n=25):
    """
    Compare n-gram representations across multiple texts.

    Args:
        texts (list): List of text responses to compare
        model_names (list): Names of models corresponding to responses
        n (int): Size of n-grams (1 for unigrams, 2 for bigrams, etc.)
        top_n (int): Number of top n-grams to consider

    Returns:
        dict: N-gram analysis results
    """
    # Initialize the results dictionary
    result = {
        "models": model_names,
        "ngram_size": n,
        "important_ngrams": {},
        "ngram_count_matrix": {},
        "differential_ngrams": []
    }

    # Make sure we have texts to analyze
    if not texts or len(texts) < 1:
        return result

    # Convert n to integer if it's a string
    if isinstance(n, str):
        n = int(n)
    
    # Convert top_n to integer if necessary
    if isinstance(top_n, str):
        top_n = int(top_n)

    try:
        # Create n-gram representations using CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=(n, n),  # Use the specified n-gram size
            max_features=1000,
            stop_words='english'
        )
        
        # Ensure each text is a string, without attempting complex preprocessing
        processed_texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        X = vectorizer.fit_transform(processed_texts)

        # Get feature names (n-grams)
        feature_names = vectorizer.get_feature_names_out()

        # Create n-gram count matrix
        ngram_counts = {}
        for i, model in enumerate(model_names):
            counts = X[i].toarray()[0]
            ngram_counts[model] = {}

            # Store n-gram frequencies for this model
            for j, ngram in enumerate(feature_names):
                if counts[j] > 0:  # Only store n-grams that appear
                    ngram_counts[model][ngram] = int(counts[j])

                    # Add to n-gram count matrix
                    if ngram not in result["ngram_count_matrix"]:
                        result["ngram_count_matrix"][ngram] = {}
                    result["ngram_count_matrix"][ngram][model] = int(counts[j])

        # Find important n-grams for each model
        for model, ngram_freq in ngram_counts.items():
            # Sort by frequency
            sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)

            # Store top N n-grams
            result["important_ngrams"][model] = [
                {"ngram": ngram, "count": count}
                for ngram, count in sorted_ngrams[:top_n]
            ]

        # Calculate differential n-grams (n-grams with biggest frequency difference between models)
        if len(model_names) >= 2:
            model1, model2 = model_names[0], model_names[1]

            # Calculate differences
            diff_scores = {}
            for ngram in result["ngram_count_matrix"]:
                count1 = result["ngram_count_matrix"][ngram].get(model1, 0)
                count2 = result["ngram_count_matrix"][ngram].get(model2, 0)

                # Absolute difference
                diff_scores[ngram] = abs(count1 - count2)

            # Sort by difference
            sorted_diffs = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)
            result["differential_ngrams"] = [ngram for ngram, _ in sorted_diffs[:top_n]]

            # Calculate overlap statistics
            model1_ngrams = set(ngram_counts.get(model1, {}).keys())
            model2_ngrams = set(ngram_counts.get(model2, {}).keys())
            common_ngrams = model1_ngrams.intersection(model2_ngrams)

            # Initialize comparisons if needed
            if "comparisons" not in result:
                result["comparisons"] = {}

            comparison_key = f"{model1} vs {model2}"
            result["comparisons"][comparison_key] = {
                "common_ngram_count": len(common_ngrams)
            }

        return result
    except Exception as e:
        import traceback
        error_msg = f"N-gram analysis error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Return basic structure with error
        return {
            "models": model_names,
            "ngram_size": n,
            "error": str(e)
        }

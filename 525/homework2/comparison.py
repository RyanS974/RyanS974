#!/usr/bin/env python3
# comparison.py

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tabulate import tabulate

# Initialize a global variable to store results
results = {}

# Function to load pre-trained embeddings
def load_pretrained_embeddings(file_path, binary=False):
    """
    Load pre-trained word embeddings from file
    
    Args:
        file_path (str): Path to the pre-trained embeddings file
        binary (bool): Whether the file is in binary format
        
    Returns:
        KeyedVectors: Loaded word embeddings
    """
    print(f"Loading pre-trained embeddings from {file_path}...")
    return KeyedVectors.load_word2vec_format(file_path, binary=binary)

# Function to find most similar words
def find_most_similar(model, word, n=10):
    """
    Find n most similar words to a given word
    
    Args:
        model: Word embedding model
        word (str): Target word
        n (int): Number of similar words to return
        
    Returns:
        list: List of tuples (word, similarity)
    """
    try:
        return model.most_similar(positive=[word], topn=n)
    except KeyError:
        return [("Word not in vocabulary", 0)]

# Function to perform vector arithmetic
def vector_arithmetic(model, positive_words, negative_words=None, n=1):
    """
    Perform vector arithmetic (e.g., king - man + woman)
    
    Args:
        model: Word embedding model
        positive_words (list): List of words to add
        negative_words (list): List of words to subtract
        n (int): Number of results to return
        
    Returns:
        list: List of tuples (word, similarity)
    """
    try:
        if negative_words is None:
            negative_words = []
        return model.most_similar(positive=positive_words, negative=negative_words, topn=n)
    except KeyError:
        return [("One or more words not in vocabulary", 0)]

# Function to find closest word to a specific vector
def closest_to_vector(model, vector, n=1):
    """
    Find the closest word to a specific vector
    
    Args:
        model: Word embedding model
        vector (numpy.ndarray): Vector to compare against
        n (int): Number of results to return
        
    Returns:
        list: List of tuples (word, similarity)
    """
    try:
        return model.similar_by_vector(vector, topn=n)
    except:
        return [("Error computing similarity", 0)]

# Function to compute similarity between two words
def word_similarity(model, word1, word2):
    """
    Compute similarity between two words
    
    Args:
        model: Word embedding model
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: Similarity score
    """
    try:
        return model.similarity(word1, word2)
    except KeyError:
        return None

# Function to find words within a specified semantic area
def semantic_field(model, words, n=5):
    """
    Find words that are similar to all words in a list (semantic field)
    
    Args:
        model: Word embedding model
        words (list): List of words defining the semantic field
        n (int): Number of results to return
        
    Returns:
        list: List of tuples (word, similarity)
    """
    try:
        return model.most_similar(positive=words, topn=n)
    except KeyError:
        return [("One or more words not in vocabulary", 0)]

# Main function to compare embeddings
def compare():
    """
    Compare different word embedding models using various queries.
    This function loads both pre-trained embeddings (e.g., GloVe, FastText)
    and your trained models (CBOW, Skip-gram) and performs a series of
    comparisons to evaluate their performance.
    """
    global results
    models = {}
    results.clear()  # Clear previous results
    
    # Attempt to load your trained models
    try:
        # Load your trained models (update paths as needed)
        models["Skip-gram"] = KeyedVectors.load("models/skipgram_model.model").wv
        models["CBOW"] = KeyedVectors.load("models/cbow_model.model").wv
        
        # Load pre-trained models
        # Download these beforehand, or update the paths
        # Example paths (update these):
        glove_path = "models/glove-wiki-gigaword-100.txt"
        fasttext_path = "models/fasttext-wiki-news-subwords-300.vec"
        
        # Update based on which pre-trained models you're using
        models["GloVe"] = load_pretrained_embeddings(glove_path, binary=False)
        models["FastText"] = load_pretrained_embeddings(fasttext_path, binary=True)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have trained or downloaded the models first.")
        return
    
    # Define your 5 queries
    
    # Query 1: Find most similar words to "computer"
    query1 = "Most similar to 'computer'"
    results[query1] = {}
    for model_name, model in models.items():
        results[query1][model_name] = find_most_similar(model, "computer", n=5)
    
    # Query 2: Vector arithmetic - country + capital relationship
    # Example: "france - paris + berlin = ?" should be close to "germany"
    query2 = "Vector arithmetic: france - paris + berlin"
    results[query2] = {}
    for model_name, model in models.items():
        results[query2][model_name] = vector_arithmetic(
            model, 
            positive_words=["france", "berlin"], 
            negative_words=["paris"], 
            n=3
        )
    
    # Query 3: Find semantic field around technology
    query3 = "Semantic field: technology, innovation, digital"
    results[query3] = {}
    for model_name, model in models.items():
        results[query3][model_name] = semantic_field(
            model, 
            ["technology", "innovation", "digital"], 
            n=5
        )
    
    # Query 4: Word similarities across domains
    query4 = "Similarity pairs"
    results[query4] = {}
    word_pairs = [
        ("doctor", "hospital"),
        ("teacher", "school"),
        ("pilot", "airplane"),
        ("chef", "restaurant"),
        ("programmer", "computer")
    ]
    for model_name, model in models.items():
        pair_results = {}
        for word1, word2 in word_pairs:
            pair_results[f"{word1}-{word2}"] = word_similarity(model, word1, word2)
        results[query4][model_name] = pair_results
    
    # Query 5: Finding analogy completions
    # Example: man is to woman as king is to ?
    query5 = "Analogies"
    results[query5] = {}
    analogies = [
        ("man", "woman", "king", "?"),
        ("good", "better", "bad", "?"),
        ("car", "road", "train", "?"),
        ("python", "programming", "english", "?"),
        ("earth", "sun", "moon", "?")
    ]
    for model_name, model in models.items():
        analogy_results = {}
        for a, b, c, _ in analogies:
            try:
                result = model.most_similar(positive=[c, b], negative=[a], topn=1)
                analogy_results[f"{a}:{b}::{c}:?"] = result[0]
            except KeyError:
                analogy_results[f"{a}:{b}::{c}:?"] = ("Word not in vocabulary", 0)
            except:
                analogy_results[f"{a}:{b}::{c}:?"] = ("Error", 0)
        results[query5][model_name] = analogy_results
    
    # Save results to CSV files for further analysis
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for query, query_results in results.items():
        if query in ["Most similar to 'computer'", "Vector arithmetic: france - paris + berlin", "Semantic field: technology, innovation, digital"]:
            # Format results for these queries
            data = []
            for model_name, model_results in query_results.items():
                for i, (word, score) in enumerate(model_results):
                    data.append({
                        "Model": model_name,
                        "Rank": i+1,
                        "Word": word,
                        "Score": score
                    })
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/{query.replace(' ', '_').replace(':', '').replace('?', '')}.csv", index=False)
        
        elif query == "Similarity pairs":
            # Format results for similarity pairs
            data = []
            for pair in word_pairs:
                pair_str = f"{pair[0]}-{pair[1]}"
                row = {"Pair": pair_str}
                for model_name in models.keys():
                    row[model_name] = query_results[model_name][pair_str]
                data.append(row)
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/similarity_pairs.csv", index=False)
        
        elif query == "Analogies":
            # Format results for analogies
            data = []
            for analogy in analogies:
                analogy_str = f"{analogy[0]}:{analogy[1]}::{analogy[2]}:?"
                row = {"Analogy": analogy_str}
                for model_name in models.keys():
                    result = query_results[model_name][analogy_str]
                    if isinstance(result, tuple):
                        row[f"{model_name}_word"] = result[0]
                        row[f"{model_name}_score"] = result[1] if isinstance(result[1], float) else None
                    else:
                        row[f"{model_name}_word"] = None
                        row[f"{model_name}_score"] = None
                data.append(row)
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/analogies.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/ directory\n")

def display_most_similar():
    query = "Most similar to 'computer'"
    if query in results:
        print(f"\n== {query} ==\n")
        for model_name, model_results in results[query].items():
            print(f"{model_name}:")
            for word, score in model_results:
                print(f"  {word}: {score:.3f}")
            print()
    else:
        print("No results available. Please run the comparison first by selecting option 4.")

def display_vector_arithmetic():
    query = "Vector arithmetic: france - paris + berlin"
    if query in results:
        print(f"\n== {query} ==\n")
        for model_name, model_results in results[query].items():
            print(f"{model_name}:")
            for word, score in model_results:
                print(f"  {word}: {score:.3f}")
            print()
    else:
        print("No results available. Please run the comparison first by selecting option 4.")

def display_semantic_field():
    query = "Semantic field: technology, innovation, digital"
    if query in results:
        print(f"\n== {query} ==\n")
        for model_name, model_results in results[query].items():
            print(f"{model_name}:")
            for word, score in model_results:
                print(f"  {word}: {score:.3f}")
            print()
    else:
        print("No results available. Please run the comparison first by selecting option 4.")

def display_word_similarity():
    query = "Similarity pairs"
    if query in results:
        print(f"\n== {query} ==\n")
        data = []
        headers = ["Pair"] + list(results[query].keys())
        for pair in [
            ("doctor", "hospital"),
            ("teacher", "school"),
            ("pilot", "airplane"),
            ("chef", "restaurant"),
            ("programmer", "computer")
        ]:
            pair_str = f"{pair[0]}-{pair[1]}"
            row = [pair_str]
            for model_name in results[query].keys():
                row.append(f"{results[query][model_name][pair_str]:.3f}" if results[query][model_name][pair_str] is not None else "N/A")
            data.append(row)
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        print("No results available. Please run the comparison first by selecting option 4.")

def display_analogies():
    query = "Analogies"
    if query in results:
        print(f"\n== {query} ==\n")
        data = []
        headers = ["Analogy"] + list(results[query].keys())
        for analogy in [
            ("man", "woman", "king", "?"),
            ("good", "better", "bad", "?"),
            ("car", "road", "train", "?"),
            ("python", "programming", "english", "?"),
            ("earth", "sun", "moon", "?")
        ]:
            analogy_str = f"{analogy[0]}:{analogy[1]}::{analogy[2]}:?"
            row = [analogy_str]
            for model_name in results[query].keys():
                result = results[query][model_name][analogy_str]
                if isinstance(result, tuple):
                    row.append(f"{result[0]} ({result[1]:.3f})" if isinstance(result[1], float) else result[0])
                else:
                    row.append("N/A")
            data.append(row)
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        print("No results available. Please run the comparison first by selecting option 4.")

# If this file is run directly, execute the compare function
if __name__ == "__main__":
    compare()
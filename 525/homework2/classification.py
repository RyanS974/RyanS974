#!/usr/bin/env python3
# classification.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
import time
import matplotlib.pyplot as plt
import pickle
import random
from datasets import load_dataset

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

# Function to prepare Wikipedia dataset for classification
def prepare_wikipedia_classification():
    """
    Prepare a classification task using the Simple English Wikipedia dataset.
    We'll create a binary classification task by categorizing articles
    based on certain keywords in their titles or content.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Preparing Wikipedia classification dataset...")
    
    # Check if we already have a processed classification dataset
    if os.path.exists('preprocessed/classification_data.pkl'):
        print("Loading pre-processed classification dataset...")
        with open('preprocessed/classification_data.pkl', 'rb') as f:
            return pickle.load(f)
    
    try:
        # First try to load the preprocessed dataset
        if os.path.exists('preprocessed/preprocessed_dataset.pkl'):
            print("Loading preprocessed Wikipedia dataset...")
            with open('preprocessed/preprocessed_dataset.pkl', 'rb') as f:
                dataset = pickle.load(f)
        else:
            # If not available, load from scratch
            print("Loading Wikipedia dataset from HuggingFace...")
            dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
        
        print("Dataset loaded. Preparing classification task...")
        
        # Extract articles and create a binary classification task
        # We'll classify articles as either science/technology related or not
        science_keywords = ['science', 'physics', 'chemistry', 'biology', 'technology', 
                          'computer', 'math', 'astronomy', 'engineering', 'medicine']
        
        texts = []
        labels = []
        
        # Limit to 5000 samples for efficiency
        sample_size = min(5000, len(dataset['train']))
        
        # Use indices of the dataset rather than the full dataset to save memory
        indices = list(range(len(dataset['train'])))
        random.seed(42)
        random.shuffle(indices)
        sample_indices = indices[:sample_size]
        
        for idx in sample_indices:
            article = dataset['train'][idx]
            
            # Get the text - use processed tokens if available, otherwise raw text
            if 'processed_tokens' in article:
                # Join tokens back into text for CountVectorizer
                text = ' '.join(article['processed_tokens'])
            else:
                text = article['text']
            
            # Determine if it's science/tech related based on title or content
            title = article.get('title', '').lower()
            is_science = 0  # Default: not science
            
            # Check title for science keywords
            if any(keyword in title for keyword in science_keywords):
                is_science = 1
            # If not in title, check first 1000 chars of content
            elif any(keyword in text.lower()[:1000] for keyword in science_keywords):
                is_science = 1
            
            texts.append(text)
            labels.append(is_science)
        
        # Ensure we have a balanced dataset
        science_count = sum(labels)
        non_science_count = len(labels) - science_count
        print(f"Initial distribution: {science_count} science articles, {non_science_count} non-science articles")
        
        # Balance the dataset if severely imbalanced
        if science_count / len(labels) < 0.3 or science_count / len(labels) > 0.7:
            print("Balancing dataset...")
            if science_count < non_science_count:
                # Downsample non-science
                non_science_indices = [i for i, label in enumerate(labels) if label == 0]
                keep_indices = random.sample(non_science_indices, science_count)
                keep_indices.extend([i for i, label in enumerate(labels) if label == 1])
            else:
                # Downsample science
                science_indices = [i for i, label in enumerate(labels) if label == 1]
                keep_indices = random.sample(science_indices, non_science_count)
                keep_indices.extend([i for i, label in enumerate(labels) if label == 0])
            
            # Filter to keep only balanced sample
            texts = [texts[i] for i in keep_indices]
            labels = [labels[i] for i in keep_indices]
            print(f"Balanced distribution: {sum(labels)} science articles, {len(labels) - sum(labels)} non-science articles")
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Classification dataset prepared: {len(X_train)} training examples, {len(X_test)} test examples")
        
        # Save the prepared dataset
        os.makedirs('preprocessed', exist_ok=True)
        with open('preprocessed/classification_data.pkl', 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        
        # Print a sample
        print("\nSample document:")
        print(X_train[0][:300] + "...")
        print(f"Label: {'Science/Tech' if y_train[0] == 1 else 'Non-Science/Tech'}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error preparing classification dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Function to train and evaluate bag-of-words model
def train_bow_model(X_train, X_test, y_train, y_test):
    """
    Train a logistic regression model with bag-of-words features.
    
    Args:
        X_train, X_test: Training and test text data
        y_train, y_test: Training and test labels
    
    Returns:
        accuracy, f1_score, training_time
    """
    print("\n=== Training Bag-of-Words Model ===")
    
    # Start timing
    start_time = time.time()
    
    # Create a pipeline with CountVectorizer and LogisticRegression
    bow_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=5000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train the model
    print("Training model...")
    bow_pipeline.fit(X_train, y_train)
    
    # End timing
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("Evaluating on test set...")
    y_pred = bow_pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature names and their coefficients
    try:
        vectorizer = bow_pipeline.named_steps['vectorizer']
        classifier = bow_pipeline.named_steps['classifier']
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for the positive class
        coefs = classifier.coef_[0]
        
        # Get top features for each class
        top_positive_idx = np.argsort(coefs)[-10:]
        top_negative_idx = np.argsort(coefs)[:10]
        
        print("\nTop features for class 0:")
        for idx in top_negative_idx:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        
        print("\nTop features for class 1:")
        for idx in top_positive_idx:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    return accuracy, f1, training_time, len(feature_names)

# Function to extract word embeddings features
def extract_embedding_features(texts, word_vectors):
    """
    Extract continuous bag-of-words features using word embeddings.
    
    Args:
        texts: List of text documents
        word_vectors: KeyedVectors model with word embeddings
    
    Returns:
        X_features: Document features as averaged word embeddings
    """
    embedding_dim = word_vectors.vector_size
    features = np.zeros((len(texts), embedding_dim))
    
    for i, text in enumerate(texts):
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Filter tokens that are in vocabulary
        valid_tokens = [token for token in tokens if token in word_vectors]
        
        if valid_tokens:
            # Get embeddings for each token and average them
            token_embeddings = [word_vectors[token] for token in valid_tokens]
            doc_embedding = np.mean(token_embeddings, axis=0)
            features[i] = doc_embedding
        else:
            # If no valid tokens, log this for debugging
            if i % 100 == 0:  # Only log occasionally to avoid flooding output
                print(f"Warning: Document {i} has no tokens in embedding vocabulary")
    
    # Check if we have any valid features
    if np.all(features == 0):
        print("Warning: No valid embeddings were found for any tokens")
        # Return a dummy feature to avoid errors
        return np.zeros((len(texts), 1))
        
    return features

# Function to train and evaluate CBOW embedding model
def train_embedding_model(X_train, X_test, y_train, y_test, embedding_path="models/fasttext-wiki-news-subwords-300.vec", binary=False):
    """
    Train a logistic regression model with continuous bag-of-words embeddings.
    
    Args:
        X_train, X_test: Training and test text data
        y_train, y_test: Training and test labels
        embedding_path: Path to pre-trained embeddings
        binary: Whether the embeddings file is in binary format
    
    Returns:
        accuracy, f1_score, training_time
    """
    print("\n=== Training Continuous Bag-of-Words Embedding Model ===")
    
    # Load pre-trained embeddings
    print(f"Loading embeddings from {embedding_path}...")
    try:
        # Try with binary=True first, then with binary=False if that fails
        try:
            word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
            print(f"Loaded embeddings (binary format) with {word_vectors.vector_size} dimensions")
        except Exception as binary_error:
            try:
                word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
                print(f"Loaded embeddings (text format) with {word_vectors.vector_size} dimensions")
            except Exception as text_error:
                # If both fail, try with the specified binary parameter
                print(f"Trying with specified binary={binary} parameter...")
                word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=binary)
                print(f"Loaded embeddings with {word_vectors.vector_size} dimensions")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Falling back to a smaller pre-trained model from gensim...")
        try:
            import gensim.downloader as api
            word_vectors = api.load("glove-wiki-gigaword-100")
            print(f"Loaded GloVe embeddings with {word_vectors.vector_size} dimensions")
        except Exception as e2:
            print(f"Error loading fallback embeddings: {e2}")
            # Return zeros to avoid division by zero in comparison
            return 0, 0, 0, 1  
    
    # Start timing
    start_time = time.time()
    
    # Extract features using embeddings
    print("Extracting features from embeddings...")
    X_train_features = extract_embedding_features(X_train, word_vectors)
    X_test_features = extract_embedding_features(X_test, word_vectors)
    
    # Check if we have valid features
    if X_train_features.shape[1] == 0:
        print("Error: No valid features extracted from embeddings")
        return 0, 0, 0, 1
    
    # Train logistic regression model
    print("Training logistic regression model...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_features, y_train)
    
    # End timing
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("Evaluating on test set...")
    y_pred = classifier.predict(X_test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, f1, training_time, word_vectors.vector_size

# Function to visualize comparison results
def visualize_comparison(bow_results, embedding_results):
    """
    Visualize comparison between BOW and embedding models.
    
    Args:
        bow_results: (accuracy, f1, training_time, num_features) for BOW model
        embedding_results: (accuracy, f1, training_time, num_features) for embedding model
    """
    # Create figure with more space to accommodate labels
    plt.figure(figsize=(18, 8))
    
    # Create subplots with specific spacing
    axs = []
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        axs.append(ax)
    
    # Model names
    models = ['Bag-of-Words', 'CBOW Embeddings']
    
    # Accuracy comparison
    axs[0].bar(models, [bow_results[0], embedding_results[0]])
    axs[0].set_title('Accuracy', fontsize=14)
    axs[0].set_ylim(0, 1)
    for i, v in enumerate([bow_results[0], embedding_results[0]]):
        axs[0].text(i, v+0.01, f"{v:.4f}", ha='center')
    
    # F1 score comparison
    axs[1].bar(models, [bow_results[1], embedding_results[1]])
    axs[1].set_title('F1 Score', fontsize=14)
    axs[1].set_ylim(0, 1)
    for i, v in enumerate([bow_results[1], embedding_results[1]]):
        axs[1].text(i, v+0.01, f"{v:.4f}", ha='center')
    
    # Training time comparison
    axs[2].bar(models, [bow_results[2], embedding_results[2]])
    axs[2].set_title('Training Time (seconds)', fontsize=14)
    for i, v in enumerate([bow_results[2], embedding_results[2]]):
        axs[2].text(i, v+0.5, f"{v:.2f}s", ha='center')
    
    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3, bottom=0.15)
    
    # Save the figure
    os.makedirs('classification_results', exist_ok=True)
    plt.savefig('classification_results/model_comparison.png', bbox_inches='tight', dpi=300)
    print("\nComparison visualization saved to 'classification_results/model_comparison.png'")
    
    # Feature dimensionality comparison
    print("\n=== Feature Dimensionality Comparison ===")
    print(f"Bag-of-Words: {bow_results[3]} features")
    print(f"CBOW Embeddings: {embedding_results[3]} features")
    
    # Additional notes on comparison
    print("\n=== Analysis ===")
    
    # Accuracy comparison
    if bow_results[0] > embedding_results[0]:
        acc_diff = bow_results[0] - embedding_results[0]
        print(f"Bag-of-Words model has higher accuracy by {acc_diff:.4f}")
    elif embedding_results[0] > bow_results[0]:
        acc_diff = embedding_results[0] - bow_results[0]
        print(f"CBOW Embeddings model has higher accuracy by {acc_diff:.4f}")
    else:
        print("Both models have the same accuracy")
    
    # Ensure we don't divide by zero
    if embedding_results[3] > 0:
        # Efficiency comparison
        dim_ratio = bow_results[3] / embedding_results[3]
        print(f"Bag-of-Words uses {dim_ratio:.1f}x more features than CBOW Embeddings")
    else:
        print("Unable to compare feature dimensions (embedding dimension is zero)")
    
    # Time comparison
    if bow_results[2] > 0 and embedding_results[2] > 0:
        time_ratio = bow_results[2] / embedding_results[2]
        if time_ratio > 1:
            print(f"Bag-of-Words was {time_ratio:.1f}x slower to train")
        else:
            time_ratio = embedding_results[2] / bow_results[2]
            print(f"CBOW Embeddings was {time_ratio:.1f}x slower to train")
    else:
        print("Unable to compare training times (one or both times are zero)")


# Main function to run text classification
def classify():
    """
    Run text classification using both bag-of-words and CBOW embedding models.
    Uses the Simple English Wikipedia dataset to create a science/tech classification task.
    """
    print("=== Text Classification with Wikipedia Dataset ===")
    
    # Prepare classification data from Wikipedia
    X_train, X_test, y_train, y_test = prepare_wikipedia_classification()
    if X_train is None:
        return
    
    # Train and evaluate bag-of-words model
    bow_results = train_bow_model(X_train, X_test, y_train, y_test)
    
    # Train and evaluate continuous bag-of-words embedding model with FastText
    # Try different embeddings paths and formats
    embedding_paths = [
        ("models/fasttext-wiki-news-subwords-300.vec", True),
        ("models/fasttext-wiki-news-subwords-300.vec", False),
        ("models/glove-wiki-gigaword-100.txt", False)
    ]
    
    # Try to load each embedding model until one succeeds
    embedding_results = None
    for path, is_binary in embedding_paths:
        if os.path.exists(path):
            print(f"Attempting to load embeddings from {path} (binary={is_binary})...")
            result = train_embedding_model(X_train, X_test, y_train, y_test, path, is_binary)
            # Check if we got valid results (non-zero feature dimension)
            if result[3] > 0:
                embedding_results = result
                print(f"Successfully used embeddings from {path}")
                break
            else:
                print(f"Failed to use embeddings from {path}")
    
    # If all explicit paths failed, try using gensim downloader
    if embedding_results is None or embedding_results[3] == 0:
        print("Attempting to load embeddings via gensim downloader...")
        try:
            import gensim.downloader as api
            # Try to download and load a smaller embedding model
            print("Downloading glove-twitter-25 embeddings (smaller model)...")
            word_vectors = api.load("glove-twitter-25")
            print(f"Loaded embeddings with {word_vectors.vector_size} dimensions")
            
            # Save to a temporary file so we can use our existing functions
            temp_path = "models/temp_embeddings.txt"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            word_vectors.save_word2vec_format(temp_path, binary=False)
            
            embedding_results = train_embedding_model(X_train, X_test, y_train, y_test, temp_path, False)
        except Exception as e:
            print(f"Error loading embeddings via downloader: {e}")
            # If still no success, create a dummy result to avoid errors
            embedding_results = (0, 0, 0, 1)
    
    # Compare and visualize results
    visualize_comparison(bow_results, embedding_results)
    
    # Save summary report
    save_summary_report(bow_results, embedding_results)


def save_summary_report(bow_results, embedding_results):
    """
    Save a summary report of the classification results.
    
    Args:
        bow_results: Results from the bag-of-words model
        embedding_results: Results from the embedding model
    """
    os.makedirs('classification_results', exist_ok=True)
    
    with open('classification_results/summary.txt', 'w') as f:
        f.write("=== Classification Results Summary ===\n\n")
        
        f.write("Bag-of-Words Model:\n")
        f.write(f"  Accuracy: {bow_results[0]:.4f}\n")
        f.write(f"  F1 Score: {bow_results[1]:.4f}\n")
        f.write(f"  Training Time: {bow_results[2]:.2f} seconds\n")
        f.write(f"  Number of Features: {bow_results[3]}\n\n")
        
        f.write("FastText CBOW Embedding Model:\n")
        f.write(f"  Accuracy: {embedding_results[0]:.4f}\n")
        f.write(f"  F1 Score: {embedding_results[1]:.4f}\n")
        f.write(f"  Training Time: {embedding_results[2]:.2f} seconds\n")
        f.write(f"  Number of Features: {embedding_results[3]}\n\n")
        
        # Comparison analysis
        f.write("Comparison Analysis:\n")
        acc_diff = abs(bow_results[0] - embedding_results[0])
        f1_diff = abs(bow_results[1] - embedding_results[1])
        
        if bow_results[0] > embedding_results[0]:
            f.write(f"  Bag-of-Words model has higher accuracy by {acc_diff:.4f}\n")
        elif embedding_results[0] > bow_results[0]:
            f.write(f"  FastText CBOW model has higher accuracy by {acc_diff:.4f}\n")
        else:
            f.write("  Both models have the same accuracy\n")
            
        if bow_results[1] > embedding_results[1]:
            f.write(f"  Bag-of-Words model has higher F1 score by {f1_diff:.4f}\n")
        elif embedding_results[1] > bow_results[1]:
            f.write(f"  FastText CBOW model has higher F1 score by {f1_diff:.4f}\n")
        else:
            f.write("  Both models have the same F1 score\n")
        
        # Feature efficiency
        dim_ratio = bow_results[3] / embedding_results[3]
        f.write(f"  Bag-of-Words uses {dim_ratio:.1f}x more features than FastText CBOW\n")
        
        # Conclusion
        f.write("\nConclusion:\n")
        if embedding_results[0] >= bow_results[0] and embedding_results[3] < bow_results[3]:
            f.write("  The FastText CBOW embedding model provides comparable or better performance\n")
            f.write("  with significantly fewer features, demonstrating the efficiency of word embeddings.\n")
        elif bow_results[0] > embedding_results[0]:
            f.write("  While the Bag-of-Words model performs better in this task, it comes at the cost\n")
            f.write("  of much higher feature dimensionality, which may be problematic for larger datasets.\n")
        else:
            f.write("  Results suggest a trade-off between performance and efficiency that should be\n")
            f.write("  considered based on the specific application requirements.\n")
    
    print(f"Summary report saved to 'classification_results/summary.txt'")

# If this file is run directly, execute the classify function
if __name__ == "__main__":
    classify()
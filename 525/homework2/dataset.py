#!/usr/bin/env python3
# dataset.py

from datasets import load_dataset
import os
from gensim.downloader import load
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def download_embeddings():
    # Paths to save the downloaded embeddings
    glove_path = "models/glove-wiki-gigaword-100.txt"
    fasttext_path = "models/fasttext-wiki-news-subwords-300.vec"
    
    # Download GloVe embeddings if not already present
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings...")
        glove_model = load("glove-wiki-gigaword-100")
        glove_model.save_word2vec_format(glove_path)
        print(f"GloVe embeddings saved to {glove_path}")
    else:
        print(f"GloVe embeddings already exist at {glove_path}")
    
    # Download Google News embeddings if not already present
    if not os.path.exists(fasttext_path):
        print("Downloading Fasttext embeddings...")
        fasttext_model = load("fasttext-wiki-news-subwords-300")
        fasttext_model.save_word2vec_format(fasttext_path, binary=True)
        print(f"Fasttext embeddings saved to {fasttext_path}")
    else:
        print(f"Fasttext embeddings already exist at {fasttext_path}")

# load the dataset
def load_data():
    print("Loading dataset...")
    dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
    print("Dataset loaded.")
    print("Example from the dataset:")
    print(dataset["train"][0]) # check the first example of the training portion of the dataset:
    return dataset

# preprocess the dataset
def preprocess_text(text):
    """
    Basic text preprocessing for word embeddings
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 1]
    return tokens

def preprocess_dataset(dataset):
    """
    Preprocess the entire Wikipedia dataset
    """
    print("Preprocessing dataset...")
    total_examples = len(dataset['train'])

    def process_example(example):
        processed_text = preprocess_text(example['text'])
        example['processed_tokens'] = processed_text
        return example
        
    # Process the dataset
        
    # Diagnostic 1: Text Length Distribution (Before Preprocessing)
    print("Calculating Diagnostic 1: Text Length Distribution (Before Preprocessing)...")
    text_lengths_before = [len(example['text']) for example in dataset['train']]
    print(f"  - Average text length before preprocessing: {sum(text_lengths_before) / len(text_lengths_before):.2f}")
    print(f"  - Minimum text length before preprocessing: {min(text_lengths_before)}")
    print(f"  - Maximum text length before preprocessing: {max(text_lengths_before)}")
    print("Diagnostic 1 completed.")

    # Diagnostic 2: Token Count Distribution (Before Preprocessing)
    print("Calculating Diagnostic 2: Token Count Distribution (Before Preprocessing)...")
    token_counts_before = [len(word_tokenize(example['text'])) for example in dataset['train']]
    print(f"  - Average token count before preprocessing: {sum(token_counts_before) / len(token_counts_before):.2f}")
    print(f"  - Minimum token count before preprocessing: {min(token_counts_before)}")
    print(f"  - Maximum token count before preprocessing: {max(token_counts_before)}")
    print("Diagnostic 2 completed.")

    print("Mapping process example function to dataset...")
    processed_dataset = dataset.map(process_example)
    print("Mapping complete.")
    
    # Diagnostic 3: Token Count Distribution (After Preprocessing)
    print("Calculating Diagnostic 3: Token Count Distribution (After Preprocessing)...")
    token_counts_after = [len(example['processed_tokens']) for example in processed_dataset['train']]
    print(f"  - Average token count after preprocessing: {sum(token_counts_after) / len(token_counts_after):.2f}")
    print(f"  - Minimum token count after preprocessing: {min(token_counts_after)}")
    print(f"  - Maximum token count after preprocessing: {max(token_counts_after)}")
    print("Diagnostic 3 completed.")

    # Diagnostic 4: Token Length Distribution (After Preprocessing)
    print("Calculating Diagnostic 4: Token Length Distribution (After Preprocessing)...")
    token_lengths = []
    for example in processed_dataset['train']:
        for token in example['processed_tokens']:
            token_lengths.append(len(token))
    print(f"  - Average token length after preprocessing: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"  - Minimum token length after preprocessing: {min(token_lengths)}")
    print(f"  - Maximum token length after preprocessing: {max(token_lengths)}")
    print("Diagnostic 4 completed.")
    
    
    # Diagnostic 5: Most Frequent Tokens (After Preprocessing)
    print("Calculating Diagnostic 5: Most Frequent Tokens (After Preprocessing)...")
    all_tokens_after = [token for example in processed_dataset['train'] for token in example['processed_tokens']]
    token_counts = pd.Series(all_tokens_after).value_counts()
    print("  - Top 20 most frequent tokens (after preprocessing):")
    for token, count in token_counts.head(20).items():
        print(f"    - {token}: {count}")
    print("Diagnostic 5 completed.")

    print("Dataset preprocessing complete.")

    # Save the preprocessed dataset
    save_preprocessed_dataset(processed_dataset, "preprocessed")
    
    return processed_dataset

def save_preprocessed_dataset(pds, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, 'preprocessed_dataset.pkl'), 'wb') as f:
        pickle.dump(pds, f)
    print(f"Preprocessed dataset saved to {directory}/preprocessed_dataset.pkl")

def load_preprocessed_dataset(directory):
    try:
        with open(os.path.join(directory, 'preprocessed_dataset.pkl'), 'rb') as f:
            print(f"Preprocessed dataset loaded from {directory}/preprocessed_dataset.pkl")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"No preprocessed dataset found in {directory}")
        return None

def preprocess_or_load_dataset(dataset, directory):
    """
    Load the preprocessed dataset if available, otherwise preprocess and save it.
    """
    pds = load_preprocessed_dataset(directory)
    if pds is None:
        if dataset is not None:
            pds = preprocess_dataset(dataset)
            save_preprocessed_dataset(pds, directory)
        else:
            print("Please load the dataset first.")
    else:
        print("Preprocessed dataset loaded successfully.")
    return pds
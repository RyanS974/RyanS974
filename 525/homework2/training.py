#!/usr/bin/env python3
# training.py

from gensim.models import Word2Vec
import os

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# train CBOW model
def train_cbow(processed_dataset, vector_size=100, window=5, min_count=5):
    """
    Train CBOW model
    
    Args:
        processed_dataset: The preprocessed dataset
        vector_size (int): Dimensionality of the word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Minimum word count threshold
        
    Returns:
        cbow_model
    """

    # Extract processed tokens
    sentences = processed_dataset['train']['processed_tokens']
    
    # Train CBOW model
    print("Training CBOW model...")
    cbow_model = Word2Vec(sentences=sentences, 
                         vector_size=vector_size,
                         window=window, 
                         min_count=min_count,
                         workers=4, 
                         sg=0)  # sg=0 for CBOW
    
    # Save the model
    save_path = 'models/cbow_model.model'
    cbow_model.save(save_path)
    print(f"CBOW model saved to {save_path}")
        
    return cbow_model

# train Skip-gram model
def train_skipgram(processed_dataset, vector_size=100, window=5, min_count=5):
    """
    Train Skip-gram model
    
    Args:
        processed_dataset: The preprocessed dataset
        vector_size (int): Dimensionality of the word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Minimum word count threshold
        
    Returns:
        skipgram_model
    """
    # Extract processed tokens
    sentences = processed_dataset['train']['processed_tokens']
    
    # Train Skip-gram model
    print("Training Skip-gram model...")
    skipgram_model = Word2Vec(sentences=sentences, 
                             vector_size=vector_size,
                             window=window, 
                             min_count=min_count,
                             workers=4, 
                             sg=1)  # sg=1 for Skip-gram
    
    # Save the model
    save_path = 'models/skipgram_model.model'
    skipgram_model.save(save_path)
    print(f"Skip-gram model saved to {save_path}")
    
    return skipgram_model

def load_model(model_path):
    """
    Load a Word2Vec model from a file
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: The loaded Word2Vec model
    """
    print(f"Loading model from {model_path}...")
    model = Word2Vec.load(model_path)
    print("Model loaded successfully.")
    return model
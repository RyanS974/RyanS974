import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import statistics
import re

def download_nltk_resources():
    """Download required NLTK resources if not already downloaded"""
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        pass

# Ensure NLTK resources are available
download_nltk_resources()

def classify_formality(text):
    """
    Classify text formality based on simple heuristics
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Formality level (Formal, Neutral, or Informal)
    """
    # Simple formality indicators
    formal_indicators = [
        r'\b(therefore|thus|consequently|furthermore|moreover|however)\b',
        r'\b(in accordance with|with respect to|regarding|concerning)\b',
        r'\b(shall|must|may|will be required to)\b',
        r'\b(it is|there are|there is)\b',
        r'\b(Mr\.|Ms\.|Dr\.|Prof\.)\b'
    ]
    
    informal_indicators = [
        r'\b(like|yeah|cool|awesome|gonna|wanna|gotta)\b',
        r'(\!{2,}|\?{2,})',
        r'\b(lol|haha|wow|omg|btw)\b',
        r'\b(don\'t|can\'t|won\'t|shouldn\'t)\b',
        r'(\.{3,})'
    ]
    
    # Calculate scores
    formal_score = sum([len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_indicators])
    informal_score = sum([len(re.findall(pattern, text, re.IGNORECASE)) for pattern in informal_indicators])
    
    # Normalize by text length
    words = len(text.split())
    if words > 0:
        formal_score = formal_score / (words / 100)  # per 100 words
        informal_score = informal_score / (words / 100)  # per 100 words
    
    # Determine formality
    if formal_score > informal_score * 1.5:
        return "Formal"
    elif informal_score > formal_score * 1.5:
        return "Informal"
    else:
        return "Neutral"

def classify_sentiment(text):
    """
    Classify text sentiment using NLTK's VADER
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment (Positive, Neutral, or Negative)
    """
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        
        if sentiment['compound'] >= 0.05:
            return "Positive"
        elif sentiment['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

def classify_complexity(text):
    """
    Classify text complexity based on sentence length and word length
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Complexity level (Simple, Average, or Complex)
    """
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return "Average"
    
    # Calculate average sentence length
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0
    
    # Calculate average word length
    words = [word for sentence in sentences for word in nltk.word_tokenize(sentence) 
             if word.isalnum()]  # only consider alphanumeric tokens
    
    avg_word_length = statistics.mean([len(word) for word in words]) if words else 0
    
    # Determine complexity
    if avg_sentence_length > 20 or avg_word_length > 6:
        return "Complex"
    elif avg_sentence_length < 12 or avg_word_length < 4:
        return "Simple"
    else:
        return "Average"

def compare_classifications(text1, text2):
    """
    Compare classifications between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        dict: Comparison results
    """
    formality1 = classify_formality(text1)
    formality2 = classify_formality(text2)
    
    sentiment1 = classify_sentiment(text1)
    sentiment2 = classify_sentiment(text2)
    
    complexity1 = classify_complexity(text1)
    complexity2 = classify_complexity(text2)
    
    results = {}
    
    if formality1 != formality2:
        results["Formality"] = f"Model 1 is {formality1.lower()}, while Model 2 is {formality2.lower()}"
    
    if sentiment1 != sentiment2:
        results["Sentiment"] = f"Model 1 has a {sentiment1.lower()} tone, while Model 2 has a {sentiment2.lower()} tone"
    
    if complexity1 != complexity2:
        results["Complexity"] = f"Model 1 uses {complexity1.lower()} language, while Model 2 uses {complexity2.lower()} language"
    
    if not results:
        results["Summary"] = "Both responses have similar writing characteristics"
    
    return results

def classify_with_roberta(text, task="sentiment", model_name=None):
    """
    Classify text using a RoBERTa model from the dataset directory
    
    Args:
        text (str): Text to analyze
        task (str): Classification task ('sentiment', 'toxicity', 'topic', 'person')
        model_name (str, optional): Specific model to use, if None will use task-appropriate model
        
    Returns:
        dict: Classification results with labels and scores
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        
        # Map tasks to appropriate pre-trained models
        task_model_map = {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment",
            "toxicity": "cardiffnlp/twitter-roberta-base-hate",
            "topic": "facebook/bart-large-mnli",  # Zero-shot classification for topics
            "person": "roberta-base"  # Default for person detection - could be fine-tuned
        }
        
        # Use mapped model if not specified
        if model_name is None and task in task_model_map:
            model_to_use = task_model_map[task]
        elif model_name is not None:
            model_to_use = model_name
        else:
            model_to_use = "roberta-base"
            
        # Special handling for zero-shot topic classification
        if task == "topic":
            classifier = pipeline("zero-shot-classification", model=model_to_use)
            topics = ["economy", "foreign policy", "healthcare", "environment", "immigration"]
            results = classifier(text, topics, multi_label=False)
            return {
                "labels": results["labels"],
                "scores": results["scores"]
            }
        else:
            # Initialize the classification pipeline
            classifier = pipeline("text-classification", model=model_to_use, return_all_scores=True)
            
            # Get classification results
            results = classifier(text)
            
            # Format results for consistent output
            if isinstance(results, list) and len(results) == 1:
                results = results[0]
                
            return {
                "task": task,
                "model": model_to_use,
                "results": results
            }
    
    except ImportError:
        return {"error": "Required packages not installed. Please install transformers and torch."}
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

def analyze_dataset_with_roberta(dataset_texts, task="topic"):
    """
    Analyze a collection of dataset texts using RoBERTa models
    
    Args:
        dataset_texts (dict): Dictionary with keys as text identifiers and values as text content
        task (str): Classification task to perform
        
    Returns:
        dict: Classification results keyed by text identifier
    """
    results = {}
    
    for text_id, text_content in dataset_texts.items():
        results[text_id] = classify_with_roberta(text_content, task=task)
    
    return results
"""
Bias detection processor for analyzing political bias in text responses
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import os
import numpy as np

# Ensure NLTK resources are available
def download_nltk_resources():
    """Download required NLTK resources if not already downloaded"""
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        pass
        
download_nltk_resources()

# Dictionary of partisan-leaning words
# These are simplified examples; a real implementation would use a more comprehensive lexicon
PARTISAN_WORDS = {
    "liberal": [
        "progressive", "equity", "climate", "reform", "collective", 
        "diversity", "inclusive", "sustainable", "justice", "regulation"
    ],
    "conservative": [
        "traditional", "freedom", "liberty", "individual", "faith", 
        "values", "efficient", "deregulation", "patriot", "security"
    ]
}

# Dictionary of framing patterns
FRAMING_PATTERNS = {
    "economic": [
        r"econom(y|ic|ics)", r"tax(es|ation)", r"budget", r"spend(ing)", 
        r"jobs?", r"wage", r"growth", r"inflation", r"invest(ment)?"
    ],
    "moral": [
        r"values?", r"ethic(s|al)", r"moral(s|ity)", r"right(s|eous)", 
        r"wrong", r"good", r"bad", r"faith", r"belief", r"tradition(al)?"
    ],
    "security": [
        r"secur(e|ity)", r"defense", r"protect(ion)?", r"threat", 
        r"danger(ous)?", r"safe(ty)?", r"nation(al)?", r"terror(ism|ist)"
    ],
    "social_welfare": [
        r"health(care)?", r"education", r"welfare", r"benefit", r"program", 
        r"help", r"assist(ance)?", r"support", r"service", r"care"
    ]
}

def detect_sentiment_bias(text):
    """
    Analyze the sentiment of a text to identify potential bias
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Sentiment analysis results
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    # Determine if sentiment indicates bias
    if sentiment['compound'] >= 0.25:
        bias_direction = "positive"
        bias_strength = min(1.0, sentiment['compound'] * 2)  # Scale to 0-1
    elif sentiment['compound'] <= -0.25:
        bias_direction = "negative"
        bias_strength = min(1.0, abs(sentiment['compound'] * 2))  # Scale to 0-1
    else:
        bias_direction = "neutral"
        bias_strength = 0.0
    
    return {
        "sentiment_scores": sentiment,
        "bias_direction": bias_direction,
        "bias_strength": bias_strength
    }

def detect_partisan_leaning(text):
    """
    Analyze text for partisan-leaning language
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Partisan leaning analysis results
    """
    text_lower = text.lower()
    
    # Count partisan words
    liberal_count = 0
    conservative_count = 0
    
    liberal_matches = []
    conservative_matches = []
    
    # Search for partisan words in text
    for word in PARTISAN_WORDS["liberal"]:
        matches = re.findall(r'\b' + word + r'\b', text_lower)
        if matches:
            liberal_count += len(matches)
            liberal_matches.extend(matches)
            
    for word in PARTISAN_WORDS["conservative"]:
        matches = re.findall(r'\b' + word + r'\b', text_lower)
        if matches:
            conservative_count += len(matches)
            conservative_matches.extend(matches)
    
    # Calculate partisan lean score (-1 to 1, negative = liberal, positive = conservative)
    total_count = liberal_count + conservative_count
    if total_count > 0:
        lean_score = (conservative_count - liberal_count) / total_count
    else:
        lean_score = 0
    
    # Determine leaning based on score
    if lean_score <= -0.2:
        leaning = "liberal"
        strength = min(1.0, abs(lean_score * 2))
    elif lean_score >= 0.2:
        leaning = "conservative"
        strength = min(1.0, lean_score * 2)
    else:
        leaning = "balanced"
        strength = 0.0
    
    return {
        "liberal_count": liberal_count,
        "conservative_count": conservative_count,
        "liberal_terms": liberal_matches,
        "conservative_terms": conservative_matches,
        "lean_score": lean_score,
        "leaning": leaning,
        "strength": strength
    }

def detect_framing_bias(text):
    """
    Analyze how the text frames issues
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Framing analysis results
    """
    text_lower = text.lower()
    framing_counts = {}
    framing_examples = {}
    
    # Count framing patterns
    for frame, patterns in FRAMING_PATTERNS.items():
        framing_counts[frame] = 0
        framing_examples[frame] = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                framing_counts[frame] += len(matches)
                # Store up to 5 examples of each frame
                unique_matches = set(matches)
                framing_examples[frame].extend(list(unique_matches)[:5])
    
    # Calculate dominant frame
    total_framing = sum(framing_counts.values())
    framing_distribution = {}
    
    if total_framing > 0:
        for frame, count in framing_counts.items():
            framing_distribution[frame] = count / total_framing
        
        dominant_frame = max(framing_counts.items(), key=lambda x: x[1])[0]
        frame_bias_strength = max(0.0, framing_distribution[dominant_frame] - 0.25)
    else:
        dominant_frame = "none"
        frame_bias_strength = 0.0
        framing_distribution = {frame: 0.0 for frame in FRAMING_PATTERNS.keys()}
    
    return {
        "framing_counts": framing_counts,
        "framing_examples": framing_examples,
        "framing_distribution": framing_distribution,
        "dominant_frame": dominant_frame,
        "frame_bias_strength": frame_bias_strength
    }

def compare_bias(text1, text2, model_names=None):
    """
    Compare potential bias in two texts
    
    Args:
        text1 (str): First text to analyze
        text2 (str): Second text to analyze
        model_names (list): Optional names of models being compared
        
    Returns:
        dict: Comparative bias analysis
    """
    # Set default model names if not provided
    if model_names is None or len(model_names) < 2:
        model_names = ["Model 1", "Model 2"]
    
    model1_name, model2_name = model_names[0], model_names[1]
    
    # Analyze each text
    sentiment_results1 = detect_sentiment_bias(text1)
    sentiment_results2 = detect_sentiment_bias(text2)
    
    partisan_results1 = detect_partisan_leaning(text1)
    partisan_results2 = detect_partisan_leaning(text2)
    
    framing_results1 = detect_framing_bias(text1)
    framing_results2 = detect_framing_bias(text2)
    
    # Determine if there's a significant difference in bias
    sentiment_difference = abs(sentiment_results1["bias_strength"] - sentiment_results2["bias_strength"])
    
    # For partisan leaning, compare the scores (negative is liberal, positive is conservative)
    partisan_difference = abs(partisan_results1["lean_score"] - partisan_results2["lean_score"])
    
    # Calculate overall bias difference
    overall_difference = (sentiment_difference + partisan_difference) / 2
    
    # Compare dominant frames
    frame_difference = framing_results1["dominant_frame"] != framing_results2["dominant_frame"] and \
                      (framing_results1["frame_bias_strength"] > 0.1 or framing_results2["frame_bias_strength"] > 0.1)
    
    # Create comparative analysis
    comparative = {
        "sentiment": {
            model1_name: sentiment_results1["bias_direction"],
            model2_name: sentiment_results2["bias_direction"],
            "difference": sentiment_difference,
            "significant": sentiment_difference > 0.3
        },
        "partisan": {
            model1_name: partisan_results1["leaning"],
            model2_name: partisan_results2["leaning"],
            "difference": partisan_difference,
            "significant": partisan_difference > 0.4
        },
        "framing": {
            model1_name: framing_results1["dominant_frame"],
            model2_name: framing_results2["dominant_frame"],
            "different_frames": frame_difference
        },
        "overall": {
            "difference": overall_difference,
            "significant_bias_difference": overall_difference > 0.35
        }
    }
    
    return {
        "models": model_names,
        model1_name: {
            "sentiment": sentiment_results1,
            "partisan": partisan_results1,
            "framing": framing_results1
        },
        model2_name: {
            "sentiment": sentiment_results2,
            "partisan": partisan_results2,
            "framing": framing_results2
        },
        "comparative": comparative
    }
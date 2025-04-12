import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import numpy as np
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Political leaning lexicons (simplified for demonstration)
# In a real implementation, these would be much more comprehensive and nuanced
LIBERAL_TERMS = {
    'progressive', 'equity', 'climate change', 'social justice', 'regulation', 'equality',
    'diversity', 'inclusion', 'workers rights', 'universal healthcare', 'welfare', 'public',
    'government program', 'marginalized', 'underrepresented', 'systemic', 'racism',
    'discrimination', 'gun control', 'green new deal', 'carbon tax', 'reproductive rights',
    'pro-choice', 'labor union', 'living wage', 'wealth tax', 'police reform'
}

CONSERVATIVE_TERMS = {
    'traditional', 'free market', 'deregulation', 'individual responsibility', 'liberty',
    'freedom', 'private sector', 'family values', 'law and order', 'tax cuts', 'limited government',
    'fiscal responsibility', 'national security', 'defense spending', 'second amendment',
    'religious freedom', 'pro-life', 'states rights', 'border security', 'merit-based',
    'job creators', 'free enterprise', 'strong military', 'patriotism', 'constitutional'
}

# Framing lexicons
ECONOMIC_FRAMING = {
    'economy', 'economic', 'cost', 'money', 'financial', 'revenue', 'tax', 'budget',
    'fiscal', 'deficit', 'inflation', 'growth', 'investment', 'market', 'trade', 'profit',
    'wage', 'income', 'gdp', 'business', 'corporation', 'industry', 'job', 'unemployment'
}

MORAL_FRAMING = {
    'moral', 'ethical', 'right', 'wrong', 'good', 'bad', 'value', 'principle', 'fair',
    'unfair', 'justice', 'dignity', 'integrity', 'honest', 'corrupt', 'compassion', 
    'respect', 'responsibility', 'duty', 'virtue', 'vice', 'sin', 'sacred', 'character'
}

SECURITY_FRAMING = {
    'security', 'safety', 'threat', 'danger', 'risk', 'fear', 'protect', 'defend',
    'attack', 'crisis', 'emergency', 'invasion', 'violence', 'crime', 'terrorism',
    'defense', 'military', 'police', 'law', 'order', 'stability', 'chaos', 'conflict'
}

def detect_sentiment(text):
    """
    Detect overall sentiment of text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment analysis results
    """
    # Use VADER for sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Classify based on compound score
    if sentiment_scores['compound'] >= 0.05:
        classification = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        classification = "Negative"
    else:
        classification = "Neutral"
    
    return {
        "compound": sentiment_scores['compound'],
        "positive": sentiment_scores['pos'],
        "neutral": sentiment_scores['neu'],
        "negative": sentiment_scores['neg'],
        "classification": classification
    }

def detect_partisan_lean(text):
    """
    Detect political leaning of text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Political leaning analysis
    """
    # Normalize text
    text_lower = text.lower()
    
    # Count occurrences of politically-charged terms
    liberal_count = 0
    conservative_count = 0
    
    for term in LIBERAL_TERMS:
        liberal_count += len(re.findall(r'\b' + term + r'\b', text_lower))
    
    for term in CONSERVATIVE_TERMS:
        conservative_count += len(re.findall(r'\b' + term + r'\b', text_lower))
    
    # Calculate total and political lean
    total_partisan_terms = liberal_count + conservative_count
    
    if total_partisan_terms > 0:
        # Scale from -1 (liberal) to 1 (conservative)
        lean_score = (conservative_count - liberal_count) / total_partisan_terms
    else:
        lean_score = 0  # Neutral if no partisan terms found
    
    # Classify based on score
    if lean_score < -0.2:
        classification = "Liberal Leaning"
    elif lean_score > 0.2:
        classification = "Conservative Leaning"
    else:
        classification = "Politically Balanced"
    
    return {
        "lean_score": lean_score,
        "liberal_terms": liberal_count,
        "conservative_terms": conservative_count,
        "total_partisan_terms": total_partisan_terms,
        "classification": classification
    }

def detect_framing_bias(text):
    """
    Detect framing bias in political context
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Framing analysis
    """
    # Normalize text
    text_lower = text.lower()
    
    # Count framing terms
    economic_count = 0
    moral_count = 0
    security_count = 0
    
    for term in ECONOMIC_FRAMING:
        economic_count += len(re.findall(r'\b' + term + r'\b', text_lower))
    
    for term in MORAL_FRAMING:
        moral_count += len(re.findall(r'\b' + term + r'\b', text_lower))
    
    for term in SECURITY_FRAMING:
        security_count += len(re.findall(r'\b' + term + r'\b', text_lower))
    
    # Calculate total framing terms
    total_framing_terms = economic_count + moral_count + security_count
    
    # Calculate percentages
    if total_framing_terms > 0:
        economic_pct = economic_count / total_framing_terms
        moral_pct = moral_count / total_framing_terms
        security_pct = security_count / total_framing_terms
    else:
        economic_pct = moral_pct = security_pct = 0
    
    # Determine primary framing
    if total_framing_terms > 0:
        max_count = max(economic_count, moral_count, security_count)
        if max_count == economic_count:
            primary_frame = "Economic"
        elif max_count == moral_count:
            primary_frame = "Moral/Ethical"
        else:
            primary_frame = "Security/Safety"
    else:
        primary_frame = "No clear framing"
    
    return {
        "economic_framing": economic_pct,
        "moral_framing": moral_pct,
        "security_framing": security_pct,
        "total_framing_terms": total_framing_terms,
        "primary_frame": primary_frame
    }

def compare_bias(texts, model_names, bias_methods=None):
    """
    Compare bias metrics across texts
    
    Args:
        texts (list): List of text responses
        model_names (list): Names of models corresponding to responses
        bias_methods (list): List of bias detection methods to apply
        
    Returns:
        dict: Comparative bias analysis
    """
    if bias_methods is None:
        bias_methods = ["Sentiment Analysis", "Partisan Leaning", "Framing Analysis"]
    
    results = {
        "models": model_names
    }
    
    # Run selected bias analyses
    if "Sentiment Analysis" in bias_methods:
        sentiment_results = {}
        for i, (text, model) in enumerate(zip(texts, model_names)):
            sentiment_results[model] = detect_sentiment(text)
        results["sentiment"] = sentiment_results
    
    if "Partisan Leaning" in bias_methods:
        partisan_results = {}
        for i, (text, model) in enumerate(zip(texts, model_names)):
            partisan_results[model] = detect_partisan_lean(text)
        results["partisan_leaning"] = partisan_results
    
    if "Framing Analysis" in bias_methods:
        framing_results = {}
        for i, (text, model) in enumerate(zip(texts, model_names)):
            framing_results[model] = detect_framing_bias(text)
        results["framing"] = framing_results
    
    # Add summary statistics
    if "Sentiment Analysis" in bias_methods:
        avg_sentiment = np.mean([results["sentiment"][model]["compound"] for model in model_names])
        sentiment_variance = np.var([results["sentiment"][model]["compound"] for model in model_names])
        results["sentiment_summary"] = {
            "average_compound": avg_sentiment,
            "variance": sentiment_variance
        }
    
    if "Partisan Leaning" in bias_methods:
        avg_lean = np.mean([results["partisan_leaning"][model]["lean_score"] for model in model_names])
        lean_variance = np.var([results["partisan_leaning"][model]["lean_score"] for model in model_names])
        results["partisan_summary"] = {
            "average_lean": avg_lean,
            "variance": lean_variance
        }
    
    return results

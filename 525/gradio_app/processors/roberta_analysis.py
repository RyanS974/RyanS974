"""
RoBERTa-based sentiment analysis for comparing LLM responses
"""
import torch
import numpy as np # ended up not using, but left in case I need it later.
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

# Global variables to store models once loaded
ROBERTA_TOKENIZER = None
ROBERTA_MODEL = None

def ensure_nltk_resources():
    """Make sure necessary NLTK resources are downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def load_roberta_model():
    """
    Load the RoBERTa model and tokenizer for sentiment analysis
    
    Returns:
        tuple: (tokenizer, model) for RoBERTa sentiment analysis
    """
    global ROBERTA_TOKENIZER, ROBERTA_MODEL
    
    # Return cached model if already loaded
    if ROBERTA_TOKENIZER is not None and ROBERTA_MODEL is not None:
        return ROBERTA_TOKENIZER, ROBERTA_MODEL
    
    print("Loading RoBERTa model and tokenizer...")
    
    try:
        # Load tokenizer and model for sentiment analysis
        ROBERTA_TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
        ROBERTA_MODEL = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
        
        return ROBERTA_TOKENIZER, ROBERTA_MODEL
    except Exception as e:
        print(f"Error loading RoBERTa model: {str(e)}")
        # Return None values if loading fails
        return None, None

def analyze_sentiment_roberta(text):
    """
    Analyze sentiment using RoBERTa model
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment analysis results with label and scores
    """
    ensure_nltk_resources()
    
    # Handle empty text
    if not text or not text.strip():
        return {
            "label": "neutral",
            "scores": {
                "contradiction": 0.33,
                "neutral": 0.34,
                "entailment": 0.33
            },
            "sentiment_score": 0.0,
            "sentence_scores": []
        }
    
    # Load model
    tokenizer, model = load_roberta_model()
    if tokenizer is None or model is None:
        return {
            "error": "Failed to load RoBERTa model",
            "label": "neutral",
            "scores": {
                "contradiction": 0.33,
                "neutral": 0.34,
                "entailment": 0.33
            },
            "sentiment_score": 0.0
        }
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Process the whole text
        encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        encoded_text = {k: v.to(device) for k, v in encoded_text.items()}
        
        with torch.no_grad():
            outputs = model(**encoded_text)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction
        contradiction_score = predictions[0, 0].item()
        neutral_score = predictions[0, 1].item()
        entailment_score = predictions[0, 2].item()
        
        # Map to sentiment
        # contradiction = negative, entailment = positive, with a scale
        sentiment_score = (entailment_score - contradiction_score) * 2  # Scale from -2 to 2
        
        # Determine sentiment label
        if sentiment_score > 0.5:
            label = "positive"
        elif sentiment_score < -0.5:
            label = "negative"
        else:
            label = "neutral"
        
        # Analyze individual sentences if text is long enough
        sentences = sent_tokenize(text)
        sentence_scores = []
        
        # Only process sentences if there are more than one and text is substantial
        if len(sentences) > 1 and len(text) > 100:
            for sentence in sentences:
                if len(sentence.split()) >= 3:  # Only analyze meaningful sentences
                    encoded_sentence = tokenizer(sentence, return_tensors='pt', truncation=True)
                    encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encoded_sentence)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # Calculate sentence sentiment score
                    sent_contradiction = predictions[0, 0].item()
                    sent_neutral = predictions[0, 1].item()
                    sent_entailment = predictions[0, 2].item()
                    sent_score = (sent_entailment - sent_contradiction) * 2
                    
                    # Determine sentiment label for this sentence
                    if sent_score > 0.5:
                        sent_label = "positive"
                    elif sent_score < -0.5:
                        sent_label = "negative"
                    else:
                        sent_label = "neutral"
                    
                    sentence_scores.append({
                        "text": sentence,
                        "score": sent_score,
                        "label": sent_label,
                        "scores": {
                            "contradiction": sent_contradiction,
                            "neutral": sent_neutral,
                            "entailment": sent_entailment
                        }
                    })
        
        return {
            "label": label,
            "scores": {
                "contradiction": contradiction_score,
                "neutral": neutral_score,
                "entailment": entailment_score
            },
            "sentiment_score": sentiment_score,
            "sentence_scores": sentence_scores
        }
    
    except Exception as e:
        import traceback
        print(f"Error analyzing sentiment with RoBERTa: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "error": str(e),
            "label": "neutral",
            "scores": {
                "contradiction": 0.33,
                "neutral": 0.34,
                "entailment": 0.33
            },
            "sentiment_score": 0.0
        }

def compare_sentiment_roberta(texts, model_names=None):
    """
    Compare sentiment between two texts using RoBERTa
    """
    print(f"Starting sentiment comparison for {len(texts)} texts")    

    # Set default model names if not provided
    if model_names is None or len(model_names) < 2:
        model_names = ["Model 1", "Model 2"]
    
    # Handle case with fewer than 2 texts
    if len(texts) < 2:
        return {
            "error": "Need at least 2 texts to compare",
            "models": model_names[:len(texts)]
        }
    
    # Get sentiment analysis for each text
    sentiment_results = []
    for text in texts:
        sentiment_results.append(analyze_sentiment_roberta(text))
    
    # Create result dictionary
    result = {
        "models": model_names[:len(texts)],
        "sentiment_analysis": {}
    }
    
    # Add individual model results
    for i, model_name in enumerate(model_names[:len(texts)]):
        result["sentiment_analysis"][model_name] = sentiment_results[i]
    
    # Compare sentiment scores
    if len(sentiment_results) >= 2:
        model1_name, model2_name = model_names[0], model_names[1]
        
        # Add null checks for the sentiment results
        score1 = 0
        score2 = 0
        
        if sentiment_results[0] and "sentiment_score" in sentiment_results[0]:
            score1 = sentiment_results[0]["sentiment_score"]
        
        if sentiment_results[1] and "sentiment_score" in sentiment_results[1]:
            score2 = sentiment_results[1]["sentiment_score"]
        
        # Calculate difference and determine which is more positive/negative
        difference = abs(score1 - score2)
        
        result["comparison"] = {
            "sentiment_difference": difference,
            "significant_difference": difference > 0.5,  # Threshold for significant difference
        }
        
        if score1 > score2:
            result["comparison"]["more_positive"] = model1_name
            result["comparison"]["more_negative"] = model2_name
            result["comparison"]["difference_direction"] = f"{model1_name} is more positive than {model2_name}"
        elif score2 > score1:
            result["comparison"]["more_positive"] = model2_name
            result["comparison"]["more_negative"] = model1_name
            result["comparison"]["difference_direction"] = f"{model2_name} is more positive than {model1_name}"
        else:
            result["comparison"]["equal_sentiment"] = True
            result["comparison"]["difference_direction"] = f"{model1_name} and {model2_name} have similar sentiment"
    
    return result

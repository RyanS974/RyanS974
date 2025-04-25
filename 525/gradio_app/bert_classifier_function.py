def classify_with_transformer(text, task="sentiment", model_name="distilbert-base-uncased"):
    """
    Classify text using a pre-trained transformer model (BERT, RoBERTa, etc.)
    
    Args:
        text (str): Text to analyze
        task (str): Classification task ('sentiment', 'emotion', etc.)
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        dict: Classification results with labels and scores
    """
    try:
        from transformers import pipeline
        
        # Map tasks to appropriate models if not specified
        task_model_map = {
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "emotion": "j-hartmann/emotion-english-distilroberta-base",
            "toxicity": "unitary/toxic-bert"
        }
        
        # Use mapped model if using default and task is in the map
        if model_name == "distilbert-base-uncased" and task in task_model_map:
            model_to_use = task_model_map[task]
        else:
            model_to_use = model_name
            
        # Initialize the classification pipeline
        classifier = pipeline(task, model=model_to_use)
        
        # Get classification results
        results = classifier(text)
        
        # Format results based on return type (list or dict)
        if isinstance(results, list):
            if len(results) == 1:
                return results[0]
            return results
        return results
    
    except ImportError:
        return {"error": "Required packages not installed. Please install transformers and torch."}
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import random

def train_classifier(texts, labels, algorithm="naive_bayes", feature_type="tfidf", cross_validation=True):
    """
    Train a classifier on labeled data
    
    Args:
        texts (list): List of text documents
        labels (list): List of corresponding labels
        algorithm (str): Classification algorithm to use
        feature_type (str): Feature extraction method
        cross_validation (bool): Whether to use cross-validation
        
    Returns:
        tuple: (model, vectorizer, metrics)
    """
    # Create vectorizer based on feature type
    if feature_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(max_features=1000)
    else:  # Bag of words
        vectorizer = CountVectorizer(max_features=1000)
    
    # Transform texts to feature vectors
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    # Select classifier based on algorithm
    if algorithm.lower() == "svm":
        model = LinearSVC()
    elif algorithm.lower() == "random_forest":
        model = RandomForestClassifier()
    elif algorithm.lower() == "logistic_regression":
        model = LogisticRegression()
    else:  # Default to Naive Bayes
        model = MultinomialNB()
    
    # Train the model
    model.fit(X, y)
    
    # Evaluate if cross-validation requested
    if cross_validation:
        cv_scores = cross_val_score(model, X, y, cv=5)
        metrics = {
            "accuracy": float(np.mean(cv_scores)),
            "cv_scores": [float(score) for score in cv_scores]
        }
    else:
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        metrics = {
            "accuracy": float(accuracy)
        }
    
    return model, vectorizer, metrics

def classify_texts(texts, model, vectorizer):
    """
    Classify texts using trained model
    
    Args:
        texts (list): List of text documents to classify
        model: Trained classifier model
        vectorizer: Fitted vectorizer
        
    Returns:
        tuple: (predictions, probabilities)
    """
    # Transform texts to feature vectors
    X = vectorizer.transform(texts)
    
    # Get predictions
    predictions = model.predict(X)
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
    else:
        # For models without predict_proba, use decision function if available
        if hasattr(model, "decision_function"):
            decision_scores = model.decision_function(X)
            # Convert to pseudo-probabilities
            if decision_scores.ndim == 1:
                # Binary classification
                probabilities = np.zeros((len(texts), 2))
                probabilities[:, 1] = 1 / (1 + np.exp(-decision_scores))
                probabilities[:, 0] = 1 - probabilities[:, 1]
            else:
                # Multiclass classification
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1)[:, np.newaxis])
                probabilities = exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis]
        else:
            # If no probability estimate available, return None
            probabilities = None
    
    return predictions, probabilities

def evaluate_classifier(predictions, ground_truth):
    """
    Evaluate classifier performance
    
    Args:
        predictions (list): Predicted labels
        ground_truth (list): True labels
        
    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Handle multi-class vs binary for other metrics
    unique_labels = np.unique(ground_truth)
    if len(unique_labels) > 2:
        # Multi-class
        precision = precision_score(ground_truth, predictions, average='weighted')
        recall = recall_score(ground_truth, predictions, average='weighted')
        f1 = f1_score(ground_truth, predictions, average='weighted')
    else:
        # Binary
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    # Convert to lists for JSON serialization
    conf_matrix_list = conf_matrix.tolist()
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix_list
    }

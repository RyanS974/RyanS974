# baselines.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
from tqdm import tqdm

def bow_representation(app_data):
    """
    Convert the dataset to a Bag-of-Words (BoW) representation.
    """
    print("\nConverting dataset to Bag-of-Words representation...")
    
    # Check if training and test sets are available
    if app_data["dataset"]["training_set"] is None or app_data["dataset"]["test_set"] is None:
        print("Error: Dataset not loaded. Please load the dataset first.")
        return app_data
    
    # Get text data from 'sms' column instead of 'text'
    train_texts = app_data["dataset"]["training_set"]["sms"]
    test_texts = app_data["dataset"]["test_set"]["sms"]
    
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(max_features=5000, min_df=2)
    
    # Fit and transform the training data
    X_train_bow = vectorizer.fit_transform(train_texts)
    
    # Transform the test data
    X_test_bow = vectorizer.transform(test_texts)
    
    # Store the vectorizer and transformed data in app_data
    app_data["baselines"]["bow_representation"] = {
        "vectorizer": vectorizer,
        "X_train": X_train_bow,
        "X_test": X_test_bow
    }
    
    print(f"Bag-of-Words representation created:")
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Training set: {X_train_bow.shape[0]} samples, {X_train_bow.shape[1]} features")
    print(f"  - Test set: {X_test_bow.shape[0]} samples, {X_test_bow.shape[1]} features")
    
    return app_data

def tfidf_representation(app_data):
    """
    Convert the dataset to a TF-IDF representation.
    """
    print("\nConverting dataset to TF-IDF representation...")
    
    # Check if training and test sets are available
    if app_data["dataset"]["training_set"] is None or app_data["dataset"]["test_set"] is None:
        print("Error: Dataset not loaded. Please load the dataset first.")
        return app_data
    
    # Get text data from 'sms' column instead of 'text'
    train_texts = app_data["dataset"]["training_set"]["sms"]
    test_texts = app_data["dataset"]["test_set"]["sms"]
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    
    # Transform the test data
    X_test_tfidf = vectorizer.transform(test_texts)
    
    # Store the vectorizer and transformed data in app_data
    app_data["baselines"]["tfidf_representation"] = {
        "vectorizer": vectorizer,
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf
    }
    
    print(f"TF-IDF representation created:")
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Training set: {X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features")
    print(f"  - Test set: {X_test_tfidf.shape[0]} samples, {X_test_tfidf.shape[1]} features")
    
    return app_data

def train_logistic_regression(app_data):
    """
    Train a logistic regression classifier using the TF-IDF representation (weighted BoW, essentially).
    """
    print("\nTraining Logistic Regression classifier (essentially a weighted BoW)...")
    
    # Check if TF-IDF representation is available
    if app_data["baselines"]["tfidf_representation"] is None:
        print("Error: TF-IDF representation not found. Please create the TF-IDF representation first.")
        return app_data
    
    # Get the TF-IDF transformed training data
    X_train_tfidf = app_data["baselines"]["tfidf_representation"]["X_train"]
    
    # Get the training labels
    y_train = app_data["dataset"]["training_set"]["label"]
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    print("Fitting logistic regression model...")
    model.fit(X_train_tfidf, y_train)
    
    # Store the model in app_data
    app_data["baselines"]["bow_model"] = model
    
    print("Logistic Regression classifier trained successfully.")
    
    return app_data

# The rest of the file remains unchanged
def test_logistic_regression(app_data):
    """
    Test the logistic regression classifier on the test set.
    """
    print("\nTesting Logistic Regression classifier...")
    
    # Check if model and TF-IDF representation are available
    if app_data["baselines"]["bow_model"] is None:
        print("Error: Model not found. Please train the model first.")
        return app_data
    
    if app_data["baselines"]["tfidf_representation"] is None:
        print("Error: TF-IDF representation not found. Please create the TF-IDF representation first.")
        return app_data
    
    # Get the model and TF-IDF transformed test data
    model = app_data["baselines"]["bow_model"]
    X_test_tfidf = app_data["baselines"]["tfidf_representation"]["X_test"]
    
    # Get the test labels
    y_test = app_data["dataset"]["test_set"]["label"]
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store the results in app_data
    app_data["baselines"]["bow_results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    print("Logistic Regression testing completed.")
    
    return app_data

def results_logistic_regression(app_data):
    """
    Display the results of the logistic regression classifier.
    """
    print("\n========== Logistic Regression (TF-IDF) Results ==========")
    
    if app_data["baselines"]["bow_results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["baselines"]["bow_results"]
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("==========================================================\n")
    
    return app_data

def train_random_baseline(app_data):
    """
    Prepare the random baseline classifier.
    """
    print("\nPreparing Random Baseline classifier...")
    
    # Check if dataset is available
    if app_data["dataset"]["training_set"] is None:
        print("Error: Training set not loaded. Please load the dataset first.")
        return app_data
    
    # Get training labels to calculate class distribution
    y_train = app_data["dataset"]["training_set"]["label"]
    
    # Calculate class distribution (proportion of each class)
    class_counts = {}
    for label in y_train:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    total_samples = len(y_train)
    class_distribution = {label: count / total_samples for label, count in class_counts.items()}
    
    # Store the class distribution in app_data
    app_data["baselines"]["random_baseline"] = {
        "class_distribution": class_distribution,
        "results": None
    }
    
    print("Random Baseline prepared successfully.")
    print(f"Class distribution: {class_distribution}")
    
    return app_data

def test_random_baseline(app_data):
    """
    Test the random baseline classifier on the test set.
    """
    print("\nTesting Random Baseline classifier...")
    
    # Check if random baseline and test set are available
    if app_data["baselines"]["random_baseline"] is None:
        print("Error: Random Baseline not prepared. Please prepare the Random Baseline first.")
        return app_data
    
    if app_data["dataset"]["test_set"] is None:
        print("Error: Test set not loaded. Please load the dataset first.")
        return app_data
    
    # Get the class distribution
    class_distribution = app_data["baselines"]["random_baseline"]["class_distribution"]
    
    # Get the test labels
    y_test = app_data["dataset"]["test_set"]["label"]
    
    # Simulate random predictions based on class distribution
    # We'll run multiple simulations and average the metrics
    num_simulations = 1000
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        # Generate random predictions
        y_pred = []
        for _ in range(len(y_test)):
            # Make a random prediction based on class distribution
            random_value = random.random()  # Random number between 0 and 1
            cumulative_prob = 0
            for label, prob in class_distribution.items():
                cumulative_prob += prob
                if random_value <= cumulative_prob:
                    y_pred.append(label)
                    break
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Handle division by zero in precision and recall
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    
    # Calculate average metrics
    avg_accuracy = sum(all_accuracy) / num_simulations
    avg_precision = sum(all_precision) / num_simulations
    avg_recall = sum(all_recall) / num_simulations
    avg_f1 = sum(all_f1) / num_simulations
    
    # Store the results in app_data
    app_data["baselines"]["random_baseline"]["results"] = {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }
    
    print("Random Baseline testing completed.")
    
    return app_data

def results_random_baseline(app_data):
    """
    Display the results of the random baseline classifier.
    """
    print("\n========== Random Baseline Results ==========")
    
    if app_data["baselines"]["random_baseline"] is None or app_data["baselines"]["random_baseline"]["results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["baselines"]["random_baseline"]["results"]
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("===========================================\n")
    
    return app_data

def train_majority_class_baseline(app_data):
    """
    Prepare the majority class baseline classifier.
    """
    print("\nPreparing Majority Class Baseline classifier...")
    
    # Check if dataset is available
    if app_data["dataset"]["training_set"] is None:
        print("Error: Training set not loaded. Please load the dataset first.")
        return app_data
    
    # Get training labels
    y_train = app_data["dataset"]["training_set"]["label"]
    
    # Count occurrences of each class
    class_counts = {}
    for label in y_train:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    # Find the majority class
    majority_class = max(class_counts, key=class_counts.get)
    
    # Store the majority class in app_data
    app_data["baselines"]["majority_class_baseline"] = {
        "majority_class": majority_class,
        "class_counts": class_counts,
        "results": None
    }
    
    print(f"Majority Class Baseline prepared successfully.")
    print(f"Majority class: {majority_class} (in this case, this is the non-spam class)")
    print(f"Class counts: {class_counts}")
    
    return app_data

def test_majority_class_baseline(app_data):
    """
    Test the majority class baseline classifier on the test set.
    """
    print("\nTesting Majority Class Baseline classifier...")
    
    # Check if majority class baseline and test set are available
    if app_data["baselines"]["majority_class_baseline"] is None:
        print("Error: Majority Class Baseline not prepared. Please prepare the Majority Class Baseline first.")
        return app_data
    
    if app_data["dataset"]["test_set"] is None:
        print("Error: Test set not loaded. Please load the dataset first.")
        return app_data
    
    # Get the majority class
    majority_class = app_data["baselines"]["majority_class_baseline"]["majority_class"]
    
    # Get the test labels
    y_test = app_data["dataset"]["test_set"]["label"]
    
    # Always predict the majority class
    y_pred = [majority_class] * len(y_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Store the results in app_data
    app_data["baselines"]["majority_class_baseline"]["results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    print("Majority Class Baseline testing completed.")
    
    return app_data

def results_majority_class_baseline(app_data):
    """
    Display the results of the majority class baseline classifier.
    """
    print("\n========== Majority Class Baseline Results ==========")
    
    if app_data["baselines"]["majority_class_baseline"] is None or app_data["baselines"]["majority_class_baseline"]["results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["baselines"]["majority_class_baseline"]["results"]
    majority_class = app_data["baselines"]["majority_class_baseline"]["majority_class"]
    
    print(f"Majority class: {majority_class}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("====================================================\n")
    
    return app_data

def compare_baselines(app_data):
    """
    Compare the results of all baselines.
    """
    print("\n========== Baseline Comparison ==========")
    
    # Check if results are available for all baselines
    if (app_data["baselines"]["bow_results"] is None or 
        app_data["baselines"]["random_baseline"] is None or 
        app_data["baselines"]["random_baseline"]["results"] is None or 
        app_data["baselines"]["majority_class_baseline"] is None or 
        app_data["baselines"]["majority_class_baseline"]["results"] is None):
        print("Error: Results not found for one or more baselines. Please test all baselines first.")
        return app_data
    
    # Get results
    lr_results = app_data["baselines"]["bow_results"]
    random_results = app_data["baselines"]["random_baseline"]["results"]
    majority_results = app_data["baselines"]["majority_class_baseline"]["results"]
    
    # Print comparison table
    print("\n" + "-" * 80)
    print(f"| {'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} |")
    print("|" + "-" * 78 + "|")
    print(f"| {'Logistic Regression (TF-IDF)':<25} | {lr_results['accuracy']:<10.4f} | {lr_results['precision']:<10.4f} | {lr_results['recall']:<10.4f} | {lr_results['f1']:<10.4f} |")
    print(f"| {'Random Baseline':<25} | {random_results['accuracy']:<10.4f} | {random_results['precision']:<10.4f} | {random_results['recall']:<10.4f} | {random_results['f1']:<10.4f} |")
    print(f"| {'Majority Class Baseline':<25} | {majority_results['accuracy']:<10.4f} | {majority_results['precision']:<10.4f} | {majority_results['recall']:<10.4f} | {majority_results['f1']:<10.4f} |")
    print("-" * 80)
    
    # Determine the best model based on F1 score
    models = {
        "Logistic Regression (TF-IDF)": lr_results['f1'],
        "Random Baseline": random_results['f1'],
        "Majority Class Baseline": majority_results['f1']
    }
    best_model = max(models, key=models.get)
    
    print(f"\nThe best performing baseline model is: {best_model} with an F1 score of {models[best_model]:.4f}")
    
    # Additional insights
    if best_model == "Logistic Regression (TF-IDF)":
        improvement_over_majority = lr_results['f1'] - majority_results['f1']
        print(f"Logistic Regression outperforms the Majority Class Baseline by {improvement_over_majority:.4f} in F1 score")
    
    # Compare to fine-tuned and zero-shot models if available
    if app_data["fine_tune"]["comparison_results"] is not None:
        print("\nComparison with Fine-tuned Models:")
        ft_results = app_data["fine_tune"]["comparison_results"]
        best_ft_model = ft_results["winner"]
        best_ft_f1 = ft_results[best_ft_model.lower()]["f1"] if best_ft_model != "Tie" else ft_results["distilbert"]["f1"]
        
        print(f"Best Fine-tuned Model: {best_ft_model} with F1 score {best_ft_f1:.4f}")
        print(f"Difference from best baseline: {best_ft_f1 - models[best_model]:.4f}")
    
    if app_data["zero_shot"]["comparison_results"] is not None:
        print("\nComparison with Zero-shot Models:")
        zs_results = app_data["zero_shot"]["comparison_results"]
        if zs_results["winner"] == "Exaone3.5":
            best_zs_f1 = app_data["zero_shot"]["exaone"]["results"]["f1"]
        elif zs_results["winner"] == "Granite3.2":
            best_zs_f1 = app_data["zero_shot"]["granite"]["results"]["f1"]
        else:
            best_zs_f1 = app_data["zero_shot"]["exaone"]["results"]["f1"]  # Default to exaone if tie
        
        print(f"Best Zero-shot Model: {zs_results['winner']} with F1 score {best_zs_f1:.4f}")
        print(f"Difference from best baseline: {best_zs_f1 - models[best_model]:.4f}")
    
    print("\n==========================================\n")
    
    return app_data
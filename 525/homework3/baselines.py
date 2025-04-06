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
    Train a logistic regression classifier using the TF-IDF representation.
    """
    print("\nTraining Logistic Regression classifier...")
    
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
    



# ===========================================================
# extra LR features (BoW model, combo model, and comparisons)
# ===========================================================

def train_logistic_regression_bow(app_data):
    """
    Train a logistic regression classifier using pure Bag-of-Words representation.
    """
    print("\nTraining Logistic Regression classifier (pure BoW)...")
    
    # Check if BoW representation is available
    if app_data["baselines"]["bow_representation"] is None:
        print("Error: BoW representation not found. Please create the BoW representation first.")
        return app_data
    
    # Get the BoW transformed training data
    X_train_bow = app_data["baselines"]["bow_representation"]["X_train"]
    
    # Get the training labels
    y_train = app_data["dataset"]["training_set"]["label"]
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    print("Fitting pure BoW logistic regression model...")
    model.fit(X_train_bow, y_train)
    
    # Store the model in app_data
    app_data["baselines"]["pure_bow_model"] = model
    
    print("Pure BoW Logistic Regression classifier trained successfully.")
    
    return app_data


def test_logistic_regression_bow(app_data):
    """
    Test the pure BoW logistic regression classifier on the test set.
    """
    print("\nTesting pure BoW Logistic Regression classifier...")
    
    # Check if model and BoW representation are available
    if app_data["baselines"]["pure_bow_model"] is None:
        print("Error: Pure BoW model not found. Please train the model first.")
        return app_data
    
    if app_data["baselines"]["bow_representation"] is None:
        print("Error: BoW representation not found. Please create the BoW representation first.")
        return app_data
    
    # Get the model and BoW transformed test data
    model = app_data["baselines"]["pure_bow_model"]
    X_test_bow = app_data["baselines"]["bow_representation"]["X_test"]
    
    # Get the test labels
    y_test = app_data["dataset"]["test_set"]["label"]
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_bow)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store the results in app_data
    app_data["baselines"]["pure_bow_results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    print("Pure BoW Logistic Regression testing completed.")
    
    return app_data


def results_logistic_regression_bow(app_data):
    """
    Display the results of the pure BoW logistic regression classifier.
    """
    print("\n========== Pure BoW Logistic Regression Results ==========")
    
    if app_data["baselines"]["pure_bow_results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["baselines"]["pure_bow_results"]
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("==========================================================\n")
    
    return app_data


def train_logistic_regression_combination(app_data, bow_weight=0.5):
    """
    Train a logistic regression classifier using a combination of BoW and TF-IDF features.
    
    Parameters:
    -----------
    app_data : dict
        The application data dictionary
    bow_weight : float, default=0.5
        Weight given to the BoW features. TF-IDF features get (1 - bow_weight)
    """
    print(f"\nTraining Logistic Regression classifier (Combined BoW & TF-IDF, BoW weight: {bow_weight})...")
    
    # Check if both representations are available
    if app_data["baselines"]["bow_representation"] is None:
        print("Error: BoW representation not found. Please create the BoW representation first.")
        return app_data
    
    if app_data["baselines"]["tfidf_representation"] is None:
        print("Error: TF-IDF representation not found. Please create the TF-IDF representation first.")
        return app_data
    
    # Import required libraries for matrix operations
    from scipy import sparse
    
    # Get the feature matrices
    X_train_bow = app_data["baselines"]["bow_representation"]["X_train"]
    X_train_tfidf = app_data["baselines"]["tfidf_representation"]["X_train"]
    
    # Normalize the matrices if they're not already normalized
    from sklearn.preprocessing import normalize
    X_train_bow_norm = normalize(X_train_bow, norm='l2', axis=1)
    X_train_tfidf_norm = normalize(X_train_tfidf, norm='l2', axis=1)
    
    # Create weighted combination
    X_train_combined = bow_weight * X_train_bow_norm + (1 - bow_weight) * X_train_tfidf_norm
    
    # Get the training labels
    y_train = app_data["dataset"]["training_set"]["label"]
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    print("Fitting combined feature logistic regression model...")
    model.fit(X_train_combined, y_train)
    
    # Store the model and weight in app_data
    app_data["baselines"]["combined_model"] = {
        "model": model,
        "bow_weight": bow_weight
    }
    
    print("Combined BoW & TF-IDF Logistic Regression classifier trained successfully.")
    
    return app_data


def test_logistic_regression_combination(app_data):
    """
    Test the combined BoW & TF-IDF logistic regression classifier on the test set.
    """
    print("\nTesting combined BoW & TF-IDF Logistic Regression classifier...")
    
    # Check if model and representations are available
    if app_data["baselines"]["combined_model"] is None:
        print("Error: Combined model not found. Please train the model first.")
        return app_data
    
    if app_data["baselines"]["bow_representation"] is None or app_data["baselines"]["tfidf_representation"] is None:
        print("Error: BoW or TF-IDF representation not found. Please create both representations first.")
        return app_data
    
    # Import required libraries
    from scipy import sparse
    from sklearn.preprocessing import normalize
    
    # Get the model and weight
    model = app_data["baselines"]["combined_model"]["model"]
    bow_weight = app_data["baselines"]["combined_model"]["bow_weight"]
    
    # Get the test feature matrices
    X_test_bow = app_data["baselines"]["bow_representation"]["X_test"]
    X_test_tfidf = app_data["baselines"]["tfidf_representation"]["X_test"]
    
    # Normalize the matrices
    X_test_bow_norm = normalize(X_test_bow, norm='l2', axis=1)
    X_test_tfidf_norm = normalize(X_test_tfidf, norm='l2', axis=1)
    
    # Create weighted combination
    X_test_combined = bow_weight * X_test_bow_norm + (1 - bow_weight) * X_test_tfidf_norm
    
    # Get the test labels
    y_test = app_data["dataset"]["test_set"]["label"]
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store the results in app_data
    app_data["baselines"]["combined_results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    print("Combined BoW & TF-IDF Logistic Regression testing completed.")
    
    return app_data


def results_logistic_regression_combination(app_data):
    """
    Display the results of the combined BoW & TF-IDF logistic regression classifier.
    """
    print("\n========== Combined BoW & TF-IDF Logistic Regression Results ==========")
    
    if app_data["baselines"]["combined_results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["baselines"]["combined_results"]
    bow_weight = app_data["baselines"]["combined_model"]["bow_weight"]
    
    print(f"BoW weight: {bow_weight}, TF-IDF weight: {1 - bow_weight}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("====================================================================\n")
    
    return app_data


def compare_all_logistic_regression(app_data):
    """
    Compare the results of all three logistic regression models:
    1. TF-IDF
    2. Pure BoW
    3. Combined BoW & TF-IDF
    """
    print("\n========== Logistic Regression Models Comparison ==========")
    
    # Check if results are available for all models
    missing_results = []
    
    # Check TF-IDF results (named 'bow_results' in app_data)
    if app_data["baselines"]["bow_results"] is None:
        missing_results.append("Logistic Regression (TF-IDF)")
    
    # Check pure BoW results
    if app_data["baselines"].get("pure_bow_results") is None:
        missing_results.append("Logistic Regression (BoW)")
    
    # Check combined model results
    if app_data["baselines"].get("combined_results") is None:
        missing_results.append("Logistic Regression (Combined)")
    
    # If any results are missing, inform the user
    if missing_results:
        print(f"Error: Results not found for the following models: {', '.join(missing_results)}")
        print("Please test all models first before comparing.")
        return app_data
    
    # Get results for all models
    tfidf_results = app_data["baselines"]["bow_results"]  # Note: misleading name in app_data
    bow_results = app_data["baselines"]["pure_bow_results"]
    combined_results = app_data["baselines"]["combined_results"]
    bow_weight = app_data["baselines"]["combined_model"]["bow_weight"]
    
    # Create a comparison table
    print("\n" + "-" * 90)
    print(f"| {'Model':<35} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} |")
    print("|" + "-" * 88 + "|")
    
    # Add each model to the table
    print(f"| {'Logistic Regression (TF-IDF)':<35} | {tfidf_results['accuracy']:<10.4f} | {tfidf_results['precision']:<10.4f} | {tfidf_results['recall']:<10.4f} | {tfidf_results['f1']:<10.4f} |")
    print(f"| {'Logistic Regression (BoW)':<35} | {bow_results['accuracy']:<10.4f} | {bow_results['precision']:<10.4f} | {bow_results['recall']:<10.4f} | {bow_results['f1']:<10.4f} |")
    
    model_name = f"Logistic Regression (Combined {bow_weight:.1f}BoW/{1-bow_weight:.1f}TF-IDF)"
    print(f"| {model_name:<35} | {combined_results['accuracy']:<10.4f} | {combined_results['precision']:<10.4f} | {combined_results['recall']:<10.4f} | {combined_results['f1']:<10.4f} |")
    
    print("-" * 90)
    
    # Determine the best model based on F1 score
    models = {
        "Logistic Regression (TF-IDF)": tfidf_results['f1'],
        "Logistic Regression (BoW)": bow_results['f1'],
        f"Logistic Regression (Combined {bow_weight:.1f}BoW/{1-bow_weight:.1f}TF-IDF)": combined_results['f1']
    }
    
    best_model = max(models, key=models.get)
    
    print(f"\nBest performing model: {best_model} with F1 score of {models[best_model]:.4f}")
    
    # Calculate performance differences
    best_f1 = models[best_model]
    for model_name, f1_score in models.items():
        if model_name != best_model:
            diff = best_f1 - f1_score
            print(f"{best_model} outperforms {model_name} by {diff:.4f} F1 points")
    
    # Provide insights based on the results
    if "TF-IDF" in best_model:
        print("\nInsight: The TF-IDF weighting scheme appears to provide better discrimination for this dataset.")
    elif "BoW" in best_model and "Combined" not in best_model:
        print("\nInsight: Simple term frequency (BoW) appears to be sufficient for this dataset.")
    elif "Combined" in best_model:
        print(f"\nInsight: A combination of BoW and TF-IDF (weight ratio {bow_weight:.1f}:{1-bow_weight:.1f}) provides the best performance.")
    
    # Storage in app_data (for reference in other comparisons)
    app_data["baselines"]["lr_comparison_results"] = {
        "winner": best_model,
        "tfidf": {"f1": tfidf_results["f1"]},
        "bow": {"f1": bow_results["f1"]},
        "combined": {"f1": combined_results["f1"]}
    }
    
    print("\n=======================================================\n")
    
    return app_data
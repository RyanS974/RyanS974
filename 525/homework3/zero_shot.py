# zero_shot.py

import requests
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
from tqdm import tqdm
import json
import os

def overview(app_data):
    """
    Provide an overview of zero-shot classification and its applications.
    """
    print("\n========== Zero-Shot Classification Overview ==========")
    print("Zero-shot classification allows models to classify data into categories")
    print("without explicit training on those categories. This is achieved using")
    print("pre-trained language models and carefully crafted prompts.")
    print("\nAdvantages:")
    print("- No fine-tuning required")
    print("- Can be applied to new categories without retraining")
    print("- Useful when labeled data is scarce")
    print("\nThis implementation uses two language models via Ollama API:")
    print("1. Exaone3.5 - A large language model")
    print("2. Granite3.2 - Another large language model")
    print("\nWe'll use these models to classify SMS messages as spam or not spam")
    print("using various prompting strategies.")
    print("=========================================================\n")

def find_best_prompt(app_data, model_name, num_samples=100):
    """
    Test different prompts on a sample of the training data to find the most effective one.
    
    Args:
        app_data: The application data structure
        model_name: Name of the model to use ("exaone3.5" or "granite3.2")
        num_samples: Number of samples to use for prompt testing
        
    Returns:
        The best performing prompt
    """
    print(f"\nFinding the best prompt for {model_name}...")
    
    # Make sure training data is available
    if app_data["dataset"]["training_set"] is None:
        print("Error: Training set not loaded. Please load the dataset first.")
        return None
    
    # List of prompts to try
    prompts = [
        "Is the following SMS message spam? Respond with 1 for spam, 0 for not spam: '{text}'",
        "Classify the following SMS as spam (1) or not spam (0): '{text}'",
        "Analyze this message and determine if it's spam. Reply with 1 for spam or 0 for not spam: '{text}'",
        "Is this SMS message legitimate or spam? Answer 0 for legitimate, 1 for spam: '{text}'",
        "This is an SMS message: '{text}' Is this spam? Answer with a single digit: 1 for yes, 0 for no."
    ]
    
    # Sample from training set
    train_data = app_data["dataset"]["training_set"]
    indices = random.sample(range(len(train_data)), min(num_samples, len(train_data)))
    samples = [train_data[i] for i in indices]
    
    best_prompt = None
    best_f1 = -1
    
    print(f"Testing {len(prompts)} different prompt templates on {len(samples)} training samples")
    
    for prompt_template in prompts:
        print(f"\nTesting prompt: {prompt_template}")
        predictions = []
        true_labels = []
        
        # Display header for prompt testing results
        print("\n" + "-" * 80)
        print("| {:<4} | {:<8} | {:<8} | {:<45} |".format("Idx", "Predicted", "Actual", "Response (truncated)"))
        print("|" + "-" * 78 + "|")
        
        for i, sample in enumerate(tqdm(samples, desc="Testing prompt")):
            prompt = prompt_template.format(text=sample['sms'])
            try:
                # Set stream=False to get a complete response
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": f"{model_name}:latest", "prompt": prompt, "stream": False},
                    timeout=30
                )
                
                response.raise_for_status()
                
                # Parse the response - this is the key change
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Display the response and prediction
                response_truncated = response_text[:45] + "..." if len(response_text) > 45 else response_text
                
                # Look for numbers in the response
                if '1' in response_text:
                    prediction = 1
                elif '0' in response_text:
                    prediction = 0
                else:
                    # If no clear prediction, consider it a failure
                    print(f"Warning: Unclear response for prompt: {prompt}")
                    print(f"Response: {response_text}")
                    # Skip this sample
                    continue
                
                print("| {:<4} | {:<8} | {:<8} | {:<45} |".format(
                    i, prediction, sample["label"], response_truncated
                ))
                
                predictions.append(prediction)
                true_labels.append(sample["label"])
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error with sample '{sample['sms'][:30]}...': {str(e)}")
                # Continue with the next sample
                continue
        
        # Calculate F1 score if we have predictions
        if predictions:
            f1 = f1_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            accuracy = accuracy_score(true_labels, predictions)
            
            print("\n" + "-" * 80)
            print(f"Prompt results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_prompt = prompt_template
                print(f"New best prompt found! F1={f1:.4f}")
        else:
            print("No valid predictions obtained for this prompt.")
    
    print(f"\nBest prompt for {model_name}: {best_prompt}")
    return best_prompt

def save_results(model_name, results):
    """
    Save the results to disk for persistence.
    
    Args:
        model_name: Name of the model ('exaone' or 'granite')
        results: Dictionary containing the results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the results to a JSON file
    filename = f'results/{model_name}_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {filename}")

def load_results(model_name):
    """
    Load the results from disk.
    
    Args:
        model_name: Name of the model ('exaone' or 'granite')
        
    Returns:
        Dictionary containing the results, or None if not found
    """
    filename = f'results/{model_name}_results.json'
    
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"\nResults loaded from {filename}")
    return results

def zero_shot_exaone(app_data):
    """
    Perform zero-shot classification using the Exaone3.5 model via the Ollama API.
    """
    print("\nPerforming zero-shot classification with Exaone3.5...")
    
    # Check if test set is available
    if app_data["dataset"]["test_set"] is None:
        print("Error: Test set not loaded. Please load the dataset first.")
        return app_data
    
    test_set = app_data["dataset"]["test_set"]
    
    # First find the best prompt using the training data
    best_prompt = find_best_prompt(app_data, "exaone3.5", num_samples=50)
    
    if best_prompt is None:
        print("Error: Failed to find a good prompt. Using default prompt.")
        best_prompt = "Is the following SMS message spam? Respond with 1 for spam, 0 for not spam: '{text}'"
    
    predictions = []
    true_labels = []
    
    # Stats for running totals
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    print(f"\nRunning zero-shot classification on {len(test_set)} test samples...")
    print(f"Using prompt template: {best_prompt}")
    
    # Display header for results
    print("\n" + "-" * 100)
    print("| {:<4} | {:<8} | {:<8} | {:<8} | {:<60} |".format(
        "Idx", "Predicted", "Actual", "Correct", "LLM Response (truncated)"
    ))
    print("|" + "-" * 98 + "|")
    
    for i, sample in enumerate(tqdm(test_set, desc="Classifying test data")):
        prompt = best_prompt.format(text=sample['sms'])
        try:
            # Add stream=False parameter here
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "exaone3.5:latest", "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the prediction from the response
            response_text = result.get("response", "").strip()
            response_truncated = response_text[:60] + "..." if len(response_text) > 60 else response_text
            
            if '1' in response_text:
                prediction = 1
            elif '0' in response_text:
                prediction = 0
            else:
                # If no clear prediction, default to not spam (0)
                print(f"Warning: Unclear response for prompt: {prompt}")
                print(f"Response: {response_text}")
                prediction = 0
                        
            # Update running stats
            total += 1
            is_correct = prediction == sample["label"]
            if is_correct:
                correct += 1
                
            if prediction == 1 and sample["label"] == 1:
                true_positives += 1
            elif prediction == 1 and sample["label"] == 0:
                false_positives += 1
            elif prediction == 0 and sample["label"] == 0:
                true_negatives += 1
            elif prediction == 0 and sample["label"] == 1:
                false_negatives += 1
            
            # Display the result
            print("| {:<4} | {:<8} | {:<8} | {:<8} | {:<60} |".format(
                i, prediction, sample["label"], "✓" if is_correct else "✗", response_truncated
            ))
            
            predictions.append(prediction)
            true_labels.append(sample["label"])
            
            # Display running totals every 10 samples
            if i > 0 and i % 10 == 0:
                running_accuracy = correct / total
                running_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                running_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                running_f1 = 2 * (running_precision * running_recall) / (running_precision + running_recall) if (running_precision + running_recall) > 0 else 0
                
                print("|" + "-" * 98 + "|")
                print(f"| Running totals after {total} samples:".ljust(100) + "|")
                print(f"| Correct: {correct}/{total} = {running_accuracy:.4f}".ljust(100) + "|")
                print(f"| Precision: {running_precision:.4f}, Recall: {running_recall:.4f}, F1: {running_f1:.4f}".ljust(100) + "|")
                print("|" + "-" * 98 + "|")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error with sample '{sample['sms'][:30]}...': {str(e)}")
            # Default to not spam (0) in case of error
            predictions.append(0)
            true_labels.append(sample["label"])
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions)
    
    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print(f"Total samples processed: {total}")
    print(f"Correct predictions: {correct}/{total} = {accuracy:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 80)
    
    # Create results dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_prompt": best_prompt,
        "total_samples": total,
        "correct_predictions": correct
    }
    
    # Store the results in app_data
    app_data["zero_shot"]["exaone"]["results"] = results
    
    # Save the results to disk for persistence
    save_results("exaone", results)
    
    print("\nZero-shot classification with Exaone3.5 completed.")
    
    return app_data

def results_exaone(app_data):
    """
    Display the results of zero-shot classification with Exaone3.5.
    """
    print("\n========== Exaone3.5 Zero-Shot Classification Results ==========")
    
    # Try to load results from disk if not in memory
    if app_data["zero_shot"]["exaone"]["results"] is None:
        # Try to load from disk
        loaded_results = load_results("exaone")
        if loaded_results is not None:
            app_data["zero_shot"]["exaone"]["results"] = loaded_results
        else:
            print("No results found. Please run the zero-shot classification first.")
            return app_data
    
    results = app_data["zero_shot"]["exaone"]["results"]
    
    print(f"Best prompt: {results.get('best_prompt', 'N/A')}")
    print(f"Total samples: {results.get('total_samples', 'N/A')}")
    print(f"Correct predictions: {results.get('correct_predictions', 'N/A')}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("=================================================================\n")
    
    return app_data

def zero_shot_granite(app_data):
    """
    Perform zero-shot classification using the Granite3.2 model via the Ollama API.
    """
    print("\nPerforming zero-shot classification with Granite3.2...")
    
    # Check if test set is available
    if app_data["dataset"]["test_set"] is None:
        print("Error: Test set not loaded. Please load the dataset first.")
        return app_data
    
    test_set = app_data["dataset"]["test_set"]
    
    # First find the best prompt using the training data
    best_prompt = find_best_prompt(app_data, "granite3.2", num_samples=50)
    
    if best_prompt is None:
        print("Error: Failed to find a good prompt. Using default prompt.")
        best_prompt = "Is the following SMS message spam? Respond with 1 for spam, 0 for not spam: '{text}'"
    
    predictions = []
    true_labels = []
    
    # Stats for running totals
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    print(f"\nRunning zero-shot classification on {len(test_set)} test samples...")
    print(f"Using prompt template: {best_prompt}")
    
    # Display header for results
    print("\n" + "-" * 100)
    print("| {:<4} | {:<8} | {:<8} | {:<8} | {:<60} |".format(
        "Idx", "Predicted", "Actual", "Correct", "LLM Response (truncated)"
    ))
    print("|" + "-" * 98 + "|")
    
    for i, sample in enumerate(tqdm(test_set, desc="Classifying test data")):
        prompt = best_prompt.format(text=sample['sms'])
        try:
            # Add stream=False parameter here
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "granite3.2:latest", "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the prediction from the response
            response_text = result.get("response", "").strip()
            response_truncated = response_text[:60] + "..." if len(response_text) > 60 else response_text
            
            if '1' in response_text:
                prediction = 1
            elif '0' in response_text:
                prediction = 0
            else:
                # If no clear prediction, default to not spam (0)
                print(f"Warning: Unclear response for prompt: {prompt}")
                print(f"Response: {response_text}")
                prediction = 0
                        
            # Update running stats
            total += 1
            is_correct = prediction == sample["label"]
            if is_correct:
                correct += 1
                
            if prediction == 1 and sample["label"] == 1:
                true_positives += 1
            elif prediction == 1 and sample["label"] == 0:
                false_positives += 1
            elif prediction == 0 and sample["label"] == 0:
                true_negatives += 1
            elif prediction == 0 and sample["label"] == 1:
                false_negatives += 1
            
            # Display the result
            print("| {:<4} | {:<8} | {:<8} | {:<8} | {:<60} |".format(
                i, prediction, sample["label"], "✓" if is_correct else "✗", response_truncated
            ))
            
            predictions.append(prediction)
            true_labels.append(sample["label"])
            
            # Display running totals every 10 samples
            if i > 0 and i % 10 == 0:
                running_accuracy = correct / total
                running_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                running_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                running_f1 = 2 * (running_precision * running_recall) / (running_precision + running_recall) if (running_precision + running_recall) > 0 else 0
                
                print("|" + "-" * 98 + "|")
                print(f"| Running totals after {total} samples:".ljust(100) + "|")
                print(f"| Correct: {correct}/{total} = {running_accuracy:.4f}".ljust(100) + "|")
                print(f"| Precision: {running_precision:.4f}, Recall: {running_recall:.4f}, F1: {running_f1:.4f}".ljust(100) + "|")
                print("|" + "-" * 98 + "|")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error with sample '{sample['sms'][:30]}...': {str(e)}")
            # Default to not spam (0) in case of error
            predictions.append(0)
            true_labels.append(sample["label"])
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions)
    
    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print(f"Total samples processed: {total}")
    print(f"Correct predictions: {correct}/{total} = {accuracy:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 80)
    
    # Create results dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_prompt": best_prompt,
        "total_samples": total,
        "correct_predictions": correct
    }
    
    # Store the results in app_data
    app_data["zero_shot"]["granite"]["results"] = results
    
    # Save the results to disk for persistence
    save_results("granite", results)
    
    print("\nZero-shot classification with Granite3.2 completed.")
    
    return app_data

def results_granite(app_data):
    """
    Display the results of zero-shot classification with Granite3.2.
    """
    print("\n========== Granite3.2 Zero-Shot Classification Results ==========")
    
    # Try to load results from disk if not in memory
    if app_data["zero_shot"]["granite"]["results"] is None:
        # Try to load from disk
        loaded_results = load_results("granite")
        if loaded_results is not None:
            app_data["zero_shot"]["granite"]["results"] = loaded_results
        else:
            print("No results found. Please run the zero-shot classification first.")
            return app_data
    
    results = app_data["zero_shot"]["granite"]["results"]
    
    print(f"Best prompt: {results.get('best_prompt', 'N/A')}")
    print(f"Total samples: {results.get('total_samples', 'N/A')}")
    print(f"Correct predictions: {results.get('correct_predictions', 'N/A')}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("==================================================================\n")
    
    return app_data

def compare_results(app_data):
    """
    Compare the results of Exaone3.5 and Granite3.2.
    """
    print("\n========== Comparison of Zero-Shot Classification Results ==========")
    
    # Check if results are available
    if app_data["zero_shot"]["exaone"]["results"] is None:
        # Try to load from disk
        loaded_results = load_results("exaone")
        if loaded_results is not None:
            app_data["zero_shot"]["exaone"]["results"] = loaded_results
        else:
            print("Error: Exaone3.5 results not found. Please run the zero-shot classification first.")
            return app_data
            
    if app_data["zero_shot"]["granite"]["results"] is None:
        # Try to load from disk
        loaded_results = load_results("granite")
        if loaded_results is not None:
            app_data["zero_shot"]["granite"]["results"] = loaded_results
        else:
            print("Error: Granite3.2 results not found. Please run the zero-shot classification first.")
            return app_data
    
    exaone_results = app_data["zero_shot"]["exaone"]["results"]
    granite_results = app_data["zero_shot"]["granite"]["results"]
    
    # Print comparison table
    print("\n" + "-" * 80)
    print(f"| {'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} |")
    print("|" + "-" * 78 + "|")
    print(f"| {'Exaone3.5':<15} | {exaone_results['accuracy']:<10.4f} | {exaone_results['precision']:<10.4f} | {exaone_results['recall']:<10.4f} | {exaone_results['f1']:<10.4f} |")
    print(f"| {'Granite3.2':<15} | {granite_results['accuracy']:<10.4f} | {granite_results['precision']:<10.4f} | {granite_results['recall']:<10.4f} | {granite_results['f1']:<10.4f} |")
    print("-" * 80)
    
    # Calculate differences and determine the winner
    accuracy_diff = exaone_results['accuracy'] - granite_results['accuracy']
    precision_diff = exaone_results['precision'] - granite_results['precision']
    recall_diff = exaone_results['recall'] - granite_results['recall']
    f1_diff = exaone_results['f1'] - granite_results['f1']
    
    # Determine the winner based on F1 score
    if f1_diff > 0:
        winner = "Exaone3.5"
        print(f"\nExaone3.5 performed better in terms of F1 score by {abs(f1_diff):.4f}")
    elif f1_diff < 0:
        winner = "Granite3.2"
        print(f"\nGranite3.2 performed better in terms of F1 score by {abs(f1_diff):.4f}")
    else:
        winner = "Tie"
        print("\nBoth models performed equally in terms of F1 score")
    
    # Compare other metrics
    if accuracy_diff > 0:
        print(f"Exaone3.5 achieved higher accuracy by {abs(accuracy_diff):.4f}")
    elif accuracy_diff < 0:
        print(f"Granite3.2 achieved higher accuracy by {abs(accuracy_diff):.4f}")
    else:
        print("Both models achieved the same accuracy")
    
    if precision_diff > 0:
        print(f"Exaone3.5 has higher precision by {abs(precision_diff):.4f} (fewer false positives)")
    elif precision_diff < 0:
        print(f"Granite3.2 has higher precision by {abs(precision_diff):.4f} (fewer false positives)")
    else:
        print("Both models have the same precision")
    
    if recall_diff > 0:
        print(f"Exaone3.5 has higher recall by {abs(recall_diff):.4f} (fewer false negatives)")
    elif recall_diff < 0:
        print(f"Granite3.2 has higher recall by {abs(recall_diff):.4f} (fewer false negatives)")
    else:
        print("Both models have the same recall")
    
    # Prompt comparison
    print("\nPrompt Comparison:")
    print(f"Exaone3.5 best prompt: {exaone_results.get('best_prompt', 'N/A')}")
    print(f"Granite3.2 best prompt: {granite_results.get('best_prompt', 'N/A')}")
    
    # Create comparison results dictionary
    comparison_results = {
        "accuracy_difference": accuracy_diff,
        "precision_difference": precision_diff,
        "recall_difference": recall_diff,
        "f1_score_difference": f1_diff,
        "winner": winner
    }
    
    # Save comparison results
    app_data["zero_shot"]["comparison_results"] = comparison_results
    
    # Save to disk
    save_results("comparison", comparison_results)
    
    print("\n====================================================================\n")
    
    return app_data

# testing method
def alternative_prompt_implementation(app_data, model_name="exaone3.5", method="classification"):
    """
    test
    
    Args:
        app_data: The application data structure
        model_name: Name of the model to use
        method: Prompting method to use: "classification", "explanation", "stepwise"
        
    Returns:
        Updated app_data
    """
    print(f"\nTrying alternative prompting method: {method}")
    
    test_set = app_data["dataset"]["test_set"]
    predictions = []
    true_labels = []
    
    for sample in tqdm(test_set[:10], desc=f"Testing {method} method"):  # Just testing on 10 samples
        if method == "classification":
            prompt = f"Task: Classify the following SMS message as spam or not spam.\n\nMessage: '{sample['sms']}'\n\nProvide only a numeric response: 1 for spam, 0 for not spam."
        elif method == "explanation":
            prompt = f"Let's examine this SMS message: '{sample['sms']}'\n\nSpam messages often have characteristics like promotional content, urgent calls to action, or requests for personal information.\n\nBased on these criteria, is this message spam? Respond with 1 for spam, 0 for not spam."
        elif method == "stepwise":
            prompt = f"Let's analyze this SMS message step by step:\n\nMessage: '{sample['sms']}'\n\nStep 1: Does it contain promotional content? (Yes/No)\nStep 2: Does it request personal information? (Yes/No)\nStep 3: Does it have an urgent call to action? (Yes/No)\nStep 4: Final decision: Is this message spam? Answer only with 1 for spam, 0 for not spam."
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": f"{model_name}:latest", "prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the prediction
            response_text = result.get("response", "").strip()
            print(f"Sample: {sample['sms'][:30]}...")
            print(f"Response: {response_text}")
            
            if '1' in response_text:
                prediction = 1
            else:
                prediction = 0
            
            predictions.append(prediction)
            true_labels.append(sample["label"])
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
    # Calculate metrics if we have predictions
    if predictions:
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        print(f"Results for {method} method: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return app_data
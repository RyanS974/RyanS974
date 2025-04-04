# fine_tune.py

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

def fine_tune_distilbert(app_data):
    """
    Fine-tune the DistilBERT model on the SMS Spam Detection dataset.
    
    Using:
    - Model: distilbert-base-uncased (already a distilled version of BERT)
    - Learning rate: 2e-5 (small value to fine-tune pre-trained weights at a slower rate)
    - Batch size: 16 (moderate batch size for good training speed)
    - Epochs: 3 
    - Optimizer: AdamW with weight decay of 0.01 (for regularization)
    - Max sequence length: 128
    """
    print("Fine-tuning DistilBERT...")
    
    # Check if we already have a trained model
    model_path = "./models/distilbert"
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Model already exists at {model_path}. Loading instead of training.")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Save to app_data
            app_data["fine_tune"]["distilbert"]["model"] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print("Successfully loaded pre-trained model.")
            return app_data
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Will train a new model instead.")
    
    # Load the training set
    train_dataset = app_data["dataset"]["training_set"]
    
    # Print the column names to verify what we're working with
    print("Dataset columns:", train_dataset.column_names)
    
    # Initialize tokenizer - using distilbert-base-uncased which is already a distilled (smaller) version
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Prepare the dataset for training
    def preprocess_function(examples):
        # this dataset uses 'sms' for the message content
        return tokenizer(examples["sms"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    
    # Ensure the dataset is formatted correctly for training
    tokenized_train = tokenized_train.remove_columns(["sms"]) 
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_train.set_format("torch")
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("./results/distilbert", exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results/distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,  
        warmup_steps=100,    
        weight_decay=0.01,   # AdamW optimizer with weight decay
        logging_dir="./logs",
        logging_steps=50,    
        save_strategy="epoch",
        save_total_limit=1,     # Only keep the most recent checkpoint
        report_to="none",    
        optim="adamw_torch", # Explicitly specify optimizer
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    os.makedirs("./models/distilbert", exist_ok=True)
    model.save_pretrained("./models/distilbert")
    tokenizer.save_pretrained("./models/distilbert")
    
    # Save to app_data
    app_data["fine_tune"]["distilbert"]["model"] = {
        "model": model,
        "tokenizer": tokenizer
    }
    
    print("DistilBERT fine-tuning completed successfully.")
    print("The warning messages were normal and related to this being a fine-tuning process layer on top of a pre-trained model.")
    print(f"Model saved to {model_path}")
    return app_data

def test_distilbert(app_data):
    """
    Test the fine-tuned DistilBERT model on the test set.
    """
    print("Testing DistilBERT model...")
    
    # Load the test set
    test_dataset = app_data["dataset"]["test_set"]
    
    # Print column names to verify
    print("Test dataset columns:", test_dataset.column_names)
    
    # Check if model exists in app_data
    if app_data["fine_tune"]["distilbert"]["model"] is None:
        try:
            # Try loading from disk
            print("Attempting to load model from ./models/distilbert")
            model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert")
            tokenizer = AutoTokenizer.from_pretrained("./models/distilbert")
            print("Successfully loaded model from disk.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please fine-tune the model first.")
            return app_data
    else:
        # Use model from app_data
        model = app_data["fine_tune"]["distilbert"]["model"]["model"]
        tokenizer = app_data["fine_tune"]["distilbert"]["model"]["tokenizer"]
        print("Using model from app_data.")
    
    # Prepare the dataset for testing
    def preprocess_function(examples):
        return tokenizer(examples["sms"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    
    tokenized_test = tokenized_test.remove_columns(["sms"])
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")
    
    # Create a trainer for evaluation
    trainer = Trainer(
        model=model,
    )
    
    # Make predictions
    print("Making predictions...")
    outputs = trainer.predict(tokenized_test)
    predictions = outputs.predictions
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_labels = test_dataset["label"]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Calculate actual counts for correct predictions
    total_samples = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    
    # Calculate true positives, false positives, false negatives
    true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1)
    false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0)
    
    # Save results to app_data
    app_data["fine_tune"]["distilbert"]["results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    
    print("DistilBERT testing completed.")
    return app_data

def results_distilbert(app_data):
    """
    Display the results of the DistilBERT model.
    """
    print("\nDistilBERT Results:")
    
    if app_data["fine_tune"]["distilbert"]["results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["fine_tune"]["distilbert"]["results"]
    
    print(f"Accuracy:  {results['accuracy']:.4f} (correctly predicted {results['correct_predictions']} out of {results['total_samples']} samples)")
    print(f"Precision: {results['precision']:.4f} (correctly identified {results['true_positives']} spam messages out of {results['true_positives'] + results['false_positives']} predicted as spam)")
    print(f"Recall:    {results['recall']:.4f} (caught {results['true_positives']} spam messages out of {results['true_positives'] + results['false_negatives']} actual spam messages)")
    print(f"F1 Score:  {results['f1']:.4f}")
    
    return app_data

def fine_tune_t5(app_data):
    """
    Fine-tune the T5 model on the SMS Spam Detection dataset.
    
    Using:
    - Model: t5-small (smallest version of T5)
    - Learning rate: 2e-5 (small value to fine-tune pre-trained weights slowly)
    - Batch size: 8 (smaller batch size due to T5's memory requirements)
    - Epochs: 3 
    - Optimizer: AdamW with weight decay of 0.01 (for regularization)
    - Max sequence length: 128
    """
    print("Fine-tuning T5...")
    
    # Check if we already have a trained model
    model_path = "./models/t5"
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Model already exists at {model_path}. Loading instead of training.")
        try:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            
            # Save to app_data
            app_data["fine_tune"]["t5"]["model"] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print("Successfully loaded pre-trained model.")
            return app_data
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Will train a new model instead.")
    
    # Load the training set
    train_dataset = app_data["dataset"]["training_set"]
    
    # Initialize tokenizer - using t5-small which is the smallest version of T5
    # Note: If available, could also use "google/t5-efficient-tiny" which is even smaller
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Prepare the dataset for training
    def preprocess_function(examples):
        # T5 expects text in a specific format for classification
        inputs = ["classify sms: " + text for text in examples["sms"]]
        targets = ["spam" if label == 1 else "ham" for label in examples["label"]]
        
        # Using 128 max length
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            model_labels = tokenizer(targets, truncation=True, padding="max_length", max_length=4)
        
        model_inputs["labels"] = model_labels["input_ids"]
        return model_inputs
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_train.set_format("torch")
    
    # Initialize model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Create output directory if it doesn't exist
    os.makedirs("./results/t5", exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results/t5",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # T5 uses more memory, so smaller batch size
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,             # AdamW optimizer with weight decay
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,            # Only keep the most recent checkpoint
        report_to="none",
        optim="adamw_torch",           # Explicitly specify optimizer
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    os.makedirs("./models/t5", exist_ok=True)
    model.save_pretrained("./models/t5")
    tokenizer.save_pretrained("./models/t5")
    
    # Save to app_data
    app_data["fine_tune"]["t5"]["model"] = {
        "model": model,
        "tokenizer": tokenizer
    }
    
    print("T5 fine-tuning completed successfully.")
    print(f"Model saved to {model_path}")
    return app_data

def test_t5(app_data):
    """
    Test the fine-tuned T5 model on the test set.
    """
    print("Testing T5 model...")
    
    # Load the test set
    test_dataset = app_data["dataset"]["test_set"]
    
    # Check if model exists in app_data
    if app_data["fine_tune"]["t5"]["model"] is None:
        try:
            # Try loading from disk
            model = T5ForConditionalGeneration.from_pretrained("./models/t5")
            tokenizer = T5Tokenizer.from_pretrained("./models/t5")
            print("Loaded model from disk.")
        except:
            print("Error: Model not found. Please fine-tune the model first.")
            return app_data
    else:
        # Use model from app_data
        model = app_data["fine_tune"]["t5"]["model"]["model"]
        tokenizer = app_data["fine_tune"]["t5"]["model"]["tokenizer"]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Make predictions
    print("Making predictions...")
    predicted_labels = []
    true_labels = []
    
    for i, example in enumerate(test_dataset):
        input_text = "classify sms: " + example["sms"]
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids.to(device)
        
        # Generate prediction
        outputs = model.generate(input_ids, max_length=8)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Map the prediction to a label (0 for ham, 1 for spam)
        predicted_label = 1 if predicted_text.lower() == "spam" else 0
        predicted_labels.append(predicted_label)
        true_labels.append(example["label"])
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_dataset)} examples")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Calculate actual counts for correct predictions
    total_samples = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    
    # Calculate true positives, false positives, false negatives
    true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1)
    false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0)
    
    # Save results to app_data
    app_data["fine_tune"]["t5"]["results"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    
    print("T5 testing completed.")
    return app_data

def results_t5(app_data):
    """
    Display the results of the T5 model.
    """
    print("\nT5 Results:")
    
    if app_data["fine_tune"]["t5"]["results"] is None:
        print("No results found. Please test the model first.")
        return app_data
    
    results = app_data["fine_tune"]["t5"]["results"]
    
    print(f"Accuracy:  {results['accuracy']:.4f} (correctly predicted {results['correct_predictions']} out of {results['total_samples']} samples)")
    print(f"Precision: {results['precision']:.4f} (correctly identified {results['true_positives']} spam messages out of {results['true_positives'] + results['false_positives']} predicted as spam)")
    print(f"Recall:    {results['recall']:.4f} (caught {results['true_positives']} spam messages out of {results['true_positives'] + results['false_negatives']} actual spam messages)")
    print(f"F1 Score:  {results['f1']:.4f}")
    
    return app_data

def compare_results(app_data):
    """
    Compare the results of the DistilBERT and T5 models.
    """
    print("\nComparing DistilBERT and T5 Results:")
    
    # Check if results exist
    if app_data["fine_tune"]["distilbert"]["results"] is None or app_data["fine_tune"]["t5"]["results"] is None:
        print("Error: Results not found for one or both models. Please test both models first.")
        return app_data
    
    distilbert_results = app_data["fine_tune"]["distilbert"]["results"]
    t5_results = app_data["fine_tune"]["t5"]["results"]
    
    # Print comparison table with improved formatting
    print("\n" + "-" * 80)
    print("| Model      | Accuracy      | Precision     | Recall        | F1 Score |")
    print("|" + "-" * 78 + "|")
    print(f"| DistilBERT | {distilbert_results['accuracy']:.4f} ({distilbert_results['correct_predictions']}/{distilbert_results['total_samples']}) | " +
          f"{distilbert_results['precision']:.4f} ({distilbert_results['true_positives']}/{distilbert_results['true_positives'] + distilbert_results['false_positives']}) | " +
          f"{distilbert_results['recall']:.4f} ({distilbert_results['true_positives']}/{distilbert_results['true_positives'] + distilbert_results['false_negatives']}) | " +
          f"{distilbert_results['f1']:.4f} |")
    print(f"| T5         | {t5_results['accuracy']:.4f} ({t5_results['correct_predictions']}/{t5_results['total_samples']}) | " +
          f"{t5_results['precision']:.4f} ({t5_results['true_positives']}/{t5_results['true_positives'] + t5_results['false_positives']}) | " +
          f"{t5_results['recall']:.4f} ({t5_results['true_positives']}/{t5_results['true_positives'] + t5_results['false_negatives']}) | " +
          f"{t5_results['f1']:.4f} |")
    print("-" * 80)
        
    # Compare F1 scores
    if distilbert_results['f1'] > t5_results['f1']:
        print("\nDistilBERT performed better in terms of F1 score.")
        winner = "DistilBERT"
    elif t5_results['f1'] > distilbert_results['f1']:
        print("\nT5 performed better in terms of F1 score.")
        winner = "T5"
    else:
        print("\nBoth models performed equally in terms of F1 score.")
        winner = "Tie"
    
    # Compare accuracy
    if distilbert_results['accuracy'] > t5_results['accuracy']:
        print(f"DistilBERT achieved higher accuracy (correctly classified {distilbert_results['correct_predictions']} vs {t5_results['correct_predictions']} samples).")
    elif t5_results['accuracy'] > distilbert_results['accuracy']:
        print(f"T5 achieved higher accuracy (correctly classified {t5_results['correct_predictions']} vs {distilbert_results['correct_predictions']} samples).")
    else:
        print(f"Both models achieved the same accuracy ({distilbert_results['correct_predictions']} correctly classified samples).")
    
    # Compare precision and recall
    if distilbert_results['precision'] > t5_results['precision']:
        print(f"DistilBERT has higher precision (fewer false positives: {distilbert_results['false_positives']} vs {t5_results['false_positives']}).")
    elif t5_results['precision'] > distilbert_results['precision']:
        print(f"T5 has higher precision (fewer false positives: {t5_results['false_positives']} vs {distilbert_results['false_positives']}).")
    
    if distilbert_results['recall'] > t5_results['recall']:
        print(f"DistilBERT has higher recall (fewer false negatives: {distilbert_results['false_negatives']} vs {t5_results['false_negatives']}).")
    elif t5_results['recall'] > distilbert_results['recall']:
        print(f"T5 has higher recall (fewer false negatives: {t5_results['false_negatives']} vs {distilbert_results['false_negatives']}).")
    
    # Save comparison results with actual counts
    app_data["fine_tune"]["comparison_results"] = {
        "distilbert": distilbert_results,
        "t5": t5_results,
        "winner": winner
    }
    
    return app_data
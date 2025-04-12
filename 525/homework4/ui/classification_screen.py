import gradio as gr
import json
import numpy as np
import matplotlib.pyplot as plt
import io

# Import classification models
from models.classifier import train_classifier, classify_texts, evaluate_classifier

def create_classification_screen():
    """
    Create the classification screen interface
    
    Returns:
        tuple: (classifier_options, classifier_params, run_classifier_btn, classifier_output)
    """
    with gr.Column() as classification_screen:
        gr.Markdown("## Classification Analysis")
        gr.Markdown("Classify LLM responses using various algorithms to detect patterns and groupings.")
        
        # Classifier options
        with gr.Group():
            classifier_options = gr.Radio(
                choices=[
                    "Topic-based Classification",
                    "Sentiment-based Classification",
                    "Political Leaning Classification",
                    "Response Style Classification",
                    "Content Focus Classification"
                ],
                value="Topic-based Classification",
                label="Classification Type"
            )
        
        # Classifier parameters
        with gr.Accordion("Classification Parameters", open=True) as classifier_params:
            prompt_selector = gr.Dropdown(
                choices=[],  # Will be populated based on dataset
                label="Select Prompt",
                info="Which prompt's responses to classify"
            )
            
            algorithm = gr.Dropdown(
                choices=[
                    "Naive Bayes",
                    "Support Vector Machine",
                    "Random Forest",
                    "Logistic Regression"
                ],
                value="Naive Bayes",
                label="Classification Algorithm"
            )
            
            feature_type = gr.Radio(
                choices=["Bag of Words", "TF-IDF", "Word Embeddings"],
                value="TF-IDF",
                label="Feature Extraction Method"
            )
            
            cross_validation = gr.Checkbox(
                value=True,
                label="Use Cross-Validation"
            )
        
        # Run classification button
        run_classifier_btn = gr.Button("Run Classification", variant="primary")
        
        # Classification output area
        with gr.Group() as classifier_output:
            with gr.Tabs():
                with gr.Tab("Results"):
                    results_table = gr.Dataframe(
                        headers=["Model", "Predicted Class", "Confidence"],
                        label="Classification Results"
                    )
                    
                with gr.Tab("Metrics"):
                    metrics_json = gr.JSON(label="Classification Metrics")
                    
                with gr.Tab("Visualization"):
                    class_plot = gr.Plot(label="Classification Visualization")
    
    return classifier_options, classifier_params, run_classifier_btn, classifier_output

def update_classification_results(dataset, classifier_type, parameters):
    """
    Update the classification results based on selected classifier and parameters
    
    Args:
        dataset (dict): The dataset containing prompts and LLM responses
        classifier_type (str): Selected classification type
        parameters (dict): Classification parameters
        
    Returns:
        tuple: (classification_results, updated_ui_components)
    """
    if not dataset or "entries" not in dataset or not dataset["entries"]:
        empty_results = {
            "error": "No dataset provided or dataset is empty"
        }
        return empty_results, gr.update(value=[]), gr.update(value=json.dumps(empty_results, indent=2)), gr.update(value=None)
    
    # Extract the prompt from parameters
    prompt = parameters.get("prompt")
    if not prompt:
        # Use the first prompt if none selected
        prompt = dataset["entries"][0]["prompt"]
    
    # Filter entries for the selected prompt
    prompt_entries = [entry for entry in dataset["entries"] if entry["prompt"] == prompt]
    
    # Extract texts and model names
    texts = [entry["response"] for entry in prompt_entries]
    models = [entry["model"] for entry in prompt_entries]
    
    # Get classification parameters
    algorithm = parameters.get("algorithm", "Naive Bayes")
    feature_type = parameters.get("feature_type", "TF-IDF")
    cross_validation = parameters.get("cross_validation", True)
    
    try:
        # Generate synthetic labels for demonstration
        # In a real implementation, you would use actual classifications
        if classifier_type == "Topic-based Classification":
            classes = ["Economy", "Foreign Policy", "Social Issues", "Environment"]
        elif classifier_type == "Sentiment-based Classification":
            classes = ["Positive", "Neutral", "Negative"]
        elif classifier_type == "Political Leaning Classification":
            classes = ["Liberal", "Centrist", "Conservative"]
        elif classifier_type == "Response Style Classification":
            classes = ["Factual", "Opinionated", "Balanced"]
        else:  # Content Focus Classification
            classes = ["Policy", "History", "Statistics", "Principles"]
            
        # Simulate classification
        # In a real implementation, this would use actual ML models
        np.random.seed(42)  # For reproducibility
        class_indices = np.random.randint(0, len(classes), len(texts))
        confidences = np.random.uniform(0.6, 0.95, len(texts))
        
        # Format results for display
        results_data = []
        for i, (model, class_idx, conf) in enumerate(zip(models, class_indices, confidences)):
            results_data.append([model, classes[class_idx], f"{conf:.2f}"])
        
        # Create metrics
        metrics = {
            "classification_type": classifier_type,
            "algorithm": algorithm,
            "feature_type": feature_type,
            "cross_validation": cross_validation,
            "accuracy": round(np.random.uniform(0.7, 0.9), 2),
            "precision": round(np.random.uniform(0.7, 0.9), 2),
            "recall": round(np.random.uniform(0.7, 0.9), 2),
            "f1_score": round(np.random.uniform(0.7, 0.9), 2),
            "class_distribution": {cls: int(np.sum(class_indices == i)) for i, cls in enumerate(classes)}
        }
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart of class distribution
        class_counts = [metrics["class_distribution"][cls] for cls in classes]
        bars = ax.bar(classes, class_counts, color='skyblue')
        
        # Add labels and numbers on top of bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom')
        
        ax.set_title(f'Class Distribution for {classifier_type}')
        ax.set_ylabel('Number of Responses')
        ax.set_ylim(0, max(class_counts) + 1)
        
        # Return results
        classification_results = {
            "type": classifier_type,
            "results": [{"model": row[0], "class": row[1], "confidence": float(row[2])} for row in results_data],
            "metrics": metrics
        }
        
        return (
            classification_results,
            gr.update(value=results_data),
            gr.update(value=json.dumps(metrics, indent=2)),
            gr.update(value=fig)
        )
        
    except Exception as e:
        error_results = {
            "error": f"Classification failed: {str(e)}"
        }
        return error_results, gr.update(value=[]), gr.update(value=json.dumps(error_results, indent=2)), gr.update(value=None)

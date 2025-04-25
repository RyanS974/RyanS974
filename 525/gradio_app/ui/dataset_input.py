import gradio as gr
import os
from utils.text_dataset_parser import load_text_file, load_builtin_datasets

def create_dataset_input():
    """
    Create the dataset input interface with prompt, response, and model fields. 

    Returns:
        tuple: (dataset_inputs, example_dropdown, load_example_btn, create_btn, prompt, response1, model1, response2, model2)
    """
    # Get built-in text datasets
    text_datasets_dir = os.path.join("dataset")
    # Check if directory exists
    if os.path.exists(text_datasets_dir):
        # Filter out summary files when listing available datasets
        text_datasets = [file.name for file in os.scandir(text_datasets_dir) 
                        if file.is_file() and file.name.endswith(".txt") and not file.name.startswith("summary-")]
    else:
        # Create directory if it doesn't exist
        os.makedirs(text_datasets_dir, exist_ok=True)
        text_datasets = []

    with gr.Column() as dataset_inputs:
        gr.Markdown("## LLM Response Comparison Dataset")
        gr.Markdown("""
        Enter a prompt and responses from two different LLMs for comparison, 
        or load one of the built-in datasets.
        """)

        # Example dataset selection
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=text_datasets,
                value=text_datasets[0] if text_datasets else None,
                label="Built-in Datasets",
                info="Select a pre-made dataset to load"
            )
            load_example_btn = gr.Button("Load Dataset", variant="secondary")

        # User input fields
        gr.Markdown("### Create Your Own Dataset")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Enter a prompt/question...")
        with gr.Row():
            response1 = gr.Textbox(label="Response 1", lines=4, placeholder="Enter the first model's response...")
            model1 = gr.Textbox(label="Model 1", placeholder="Enter the first model's name...")
        with gr.Row():
            response2 = gr.Textbox(label="Response 2", lines=4, placeholder="Enter the second model's response...")
            model2 = gr.Textbox(label="Model 2", placeholder="Enter the second model's name...")

        create_btn = gr.Button("Create Dataset", variant="primary")

    return dataset_inputs, example_dropdown, load_example_btn, create_btn, prompt, response1, model1, response2, model2


def load_example_dataset(file_name):
    """
    Load a built-in dataset from a text file.

    Args:
        file_name (str): Name of the text file to load.

    Returns:
        tuple: Values for prompt, response1, model1, response2, and model2.
    """
    if not file_name:
        return "", "", "", "", ""
        
    file_path = os.path.join("dataset", file_name)
    if os.path.exists(file_path):
        try:
            dataset = load_text_file(file_path)
            return (
                dataset.get("prompt", ""),
                dataset.get("response1", ""),
                dataset.get("model1", ""),
                dataset.get("response2", ""),
                dataset.get("model2", "")
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return "Error loading dataset", "", "", "", ""
    else:
        print(f"File not found: {file_path}")
        return "File not found", "", "", "", ""  # Clear error message as first field


def create_user_dataset(prompt, response1, model1, response2, model2):
    """
    Create a user-defined dataset entry.

    Args:
        prompt (str): The prompt text.
        response1 (str): The first model's response.
        model1 (str): The first model's name.
        response2 (str): The second model's response.
        model2 (str): The second model's name.

    Returns:
        dict: Dataset entry with prompt, response1, model1, response2, and model2.
    """
    # Validate inputs
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not response1.strip():
        raise ValueError("Response 1 cannot be empty")
    if not response2.strip():
        raise ValueError("Response 2 cannot be empty")
        
    # Use default model names if not provided
    model1_name = model1.strip() if model1.strip() else "Model 1"
    model2_name = model2.strip() if model2.strip() else "Model 2"
    
    return {
        "prompt": prompt.strip(),
        "response1": response1.strip(),
        "model1": model1_name,
        "response2": response2.strip(),
        "model2": model2_name
    }
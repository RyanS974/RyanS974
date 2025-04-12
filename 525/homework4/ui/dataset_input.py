import gradio as gr
import json
import os

def create_dataset_input():
    """
    Create the dataset input interface with prompt, response, model fields
    
    Returns:
        tuple: (dataset_inputs, example_dropdown, load_example_btn, analyze_btn)
    """
    # Available example datasets
    example_datasets = [
        "US_Election_2024.json",
        "Climate_Policy.json",
        "Immigration_Reform.json",
        "Healthcare_Debate.json"
    ]
    
    with gr.Column() as dataset_inputs:
        gr.Markdown("## LLM Response Comparison Dataset")
        gr.Markdown("""
        Enter multiple prompt-response pairs from different LLMs on political topics. 
        You can add as many entries as you want or load one of our example datasets.
        """)
        
        # Example dataset selection
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=example_datasets,
                value=example_datasets[0],
                label="Example Datasets",
                info="Select a pre-made dataset to analyze"
            )
            load_example_btn = gr.Button("Load Example", variant="secondary")
        
        # Dataset entries
        entry_list = []
        with gr.Group() as entries_container:
            # Default three entries to start
            for i in range(3):
                with gr.Group() as entry:
                    gr.Markdown(f"### Entry {i+1}")
                    prompt = gr.Textbox(label="Prompt/Question", lines=2, placeholder="Enter a political question or prompt...")
                    model = gr.Textbox(label="LLM Model", placeholder="Enter the model name (e.g., GPT-4, Claude, Llama-2)")
                    response = gr.Textbox(label="Response", lines=4, placeholder="Paste the model's response here...")
                    entry_list.append({"prompt": prompt, "model": model, "response": response})
        
        # Add more entries
        def add_entry(entry_list):
            new_entry = {
                "prompt": gr.Textbox(label=f"Prompt/Question", lines=2, placeholder="Enter a political question or prompt..."),
                "model": gr.Textbox(label="LLM Model", placeholder="Enter the model name (e.g., GPT-4, Claude, Llama-2)"),
                "response": gr.Textbox(label="Response", lines=4, placeholder="Paste the model's response here...")
            }
            entry_list.append(new_entry)
            return entry_list
        
        add_entry_btn = gr.Button("Add Another Entry", variant="primary")
        add_entry_btn.click(fn=add_entry, inputs=[gr.State(entry_list)], outputs=[entries_container])
        
        analyze_btn = gr.Button("Analyze Responses", variant="primary", size="large")
    
    return dataset_inputs, example_dropdown, load_example_btn, analyze_btn

def process_dataset_submission(dataset_inputs):
    """
    Process the submitted dataset and prepare for analysis
    
    Args:
        dataset_inputs: Input fields from the dataset input screen
        
    Returns:
        dict: Structured dataset ready for analysis
    """
    # Extract values from input fields
    structured_dataset = {
        "entries": []
    }
    
    # Process each entry
    for i in range(0, len(dataset_inputs), 3):  # Each entry has 3 fields
        if i+2 < len(dataset_inputs) and dataset_inputs[i].value and dataset_inputs[i+1].value and dataset_inputs[i+2].value:
            entry = {
                "prompt": dataset_inputs[i].value,
                "model": dataset_inputs[i+1].value,
                "response": dataset_inputs[i+2].value
            }
            structured_dataset["entries"].append(entry)
    
    return structured_dataset

def load_example_dataset(example_name):
    """
    Load an example dataset from file
    
    Args:
        example_name (str): Name of the example dataset file
        
    Returns:
        list: Updated values for dataset input fields
    """
    # Path to example datasets
    example_path = f"examples/{example_name}"
    
    try:
        with open(example_path, 'r') as file:
            data = json.load(file)
        
        # Create list of values to update the input fields
        values = []
        for entry in data["entries"]:
            values.extend([
                entry["prompt"],
                entry["model"],
                entry["response"]
            ])
        
        return gr.update(value=values)
    
    except Exception as e:
        print(f"Error loading example dataset: {e}")
        return gr.update(value=[])

"""
Utility functions for parsing text-based dataset files for LLM response comparator.
"""
import re
from pathlib import Path

def parse_text_file(file_path):
    """
    Parse a text file to extract prompt, response1, model1, response2, and model2.
    
    Format:
    - \prompt= followed by the prompt text
    - \response1= followed by the first model's response
    - \model1= followed by the first model's name
    - \response2= followed by the second model's response
    - \model2= followed by the second model's name
    
    Args:
        file_path (str): Path to the text file.
        
    Returns:
        dict: Dictionary with prompt, response1, model1, response2, and model2.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract sections using regular expressions
    prompt = re.search(r'\\prompt=(.*?)(?=\\response1=|$)', content, re.DOTALL)
    response1 = re.search(r'\\response1=(.*?)(?=\\model1=|$)', content, re.DOTALL)
    model1 = re.search(r'\\model1=(.*?)(?=\\response2=|$)', content, re.DOTALL)
    response2 = re.search(r'\\response2=(.*?)(?=\\model2=|$)', content, re.DOTALL)
    model2 = re.search(r'\\model2=(.*?)(?=$)', content, re.DOTALL)
    
    return {
        "prompt": prompt.group(1).strip() if prompt else "",
        "response1": response1.group(1).strip() if response1 else "",
        "model1": model1.group(1).strip() if model1 else "",
        "response2": response2.group(1).strip() if response2 else "",
        "model2": model2.group(1).strip() if model2 else ""
    }

def load_text_file(file_path):
    """
    Load a single text file as a dataset entry.
    
    Args:
        file_path (str): Path to the text file.
        
    Returns:
        dict: Dataset entry with prompt, response1, model1, response2, and model2.
    """
    return parse_text_file(file_path)

def load_builtin_datasets(directory_path):
    """
    Load all built-in datasets from a directory.
    
    Args:
        directory_path (str): Path to the directory containing text files.
        
    Returns:
        list: List of dataset entries.
    """
    path = Path(directory_path)
    text_files = list(path.glob('*.txt'))
    return [parse_text_file(str(file_path)) for file_path in text_files]
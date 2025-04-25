"""
UI components for RoBERTa sentiment analysis screen
"""
import gradio as gr
from processors.roberta_analysis import compare_sentiment_roberta

def create_roberta_screen():
    """
    Create the RoBERTa sentiment analysis screen components 
    
    Returns:
        tuple: (run_roberta_btn, roberta_output, sentence_level, visualization_style, visualization_container, roberta_status)
    """
    with gr.Column() as roberta_screen:
        gr.Markdown("## RoBERTa Sentiment Analysis")
        gr.Markdown("""
        This tab uses the RoBERTa transformer model to perform sentiment analysis on the LLM responses
        and compare their emotional tones. RoBERTa was trained on a diverse dataset and can detect subtle
        differences in sentiment that simpler rule-based classifiers might miss.
        
        Click 'Run Sentiment Analysis' to analyze your dataset.
        """)
        
        with gr.Row():            
            run_roberta_btn = gr.Button("Run Sentiment Analysis", variant="primary", size="large")
        
        # Status message area
        roberta_status = gr.Markdown(visible=False)
        
        # Hidden output to store raw analysis results
        roberta_output = gr.JSON(label="Sentiment Analysis Results", visible=False)
        
        # Placeholder for visualization container - this won't be directly updated
        # but is included for backward compatibility
        visualization_container = gr.Markdown(visible=False)
        
    return run_roberta_btn, roberta_output, visualization_container, roberta_status

def process_roberta_request(dataset):
    """
    Process the RoBERTa sentiment analysis request
    
    Args:
        dataset (dict): The input dataset
        sentence_level (bool): Whether to perform sentence-level analysis
        visualization_style (str): Visualization style preference
        
    Returns:
        dict: Analysis results
    """
    if not dataset or "entries" not in dataset or not dataset["entries"]:
        return {"error": "No dataset loaded. Please create or load a dataset first."}
    
    # Initialize the results structure
    results = {"analyses": {}}
    
    # Get the prompt text from the first entry
    prompt_text = dataset["entries"][0].get("prompt", "")
    if not prompt_text:
        return {"error": "No prompt found in dataset"}
    
    # Initialize the analysis container for this prompt
    results["analyses"][prompt_text] = {}
    
    # Get model names and responses
    model1_name = dataset["entries"][0].get("model", "Model 1")
    model2_name = dataset["entries"][1].get("model", "Model 2")
    
    model1_response = dataset["entries"][0].get("response", "")
    model2_response = dataset["entries"][1].get("response", "")
    
    try:
        # Perform RoBERTa sentiment analysis
        print("Starting RoBERTa sentiment analysis...")
        
        sentiment_results = compare_sentiment_roberta(
            texts=[model1_response, model2_response],
            model_names=[model1_name, model2_name]
        )
        
        # Store the results
        results["analyses"][prompt_text]["roberta_sentiment"] = sentiment_results
        
        # Add metadata about the analysis
        results["analyses"][prompt_text]["roberta_sentiment"]["metadata"] = {
            "sentence_level": "",
            "visualization_style": ""
        }
        
        return results
    
    except Exception as e:
        import traceback
        error_msg = f"Error in RoBERTa sentiment analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Return error information
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "analyses": {
                prompt_text: {
                    "roberta_sentiment": {
                        "error": str(e),
                        "models": [model1_name, model2_name],
                        "message": "RoBERTa sentiment analysis failed. Try again or use a different analysis method."
                    }
                }
            }
        }
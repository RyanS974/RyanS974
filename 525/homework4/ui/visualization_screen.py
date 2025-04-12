import gradio as gr
import json
import matplotlib.pyplot as plt
import io
import base64

# Import visualization modules
from visualizers.topic_viz import plot_topic_distribution, create_topic_wordcloud
from visualizers.ngram_viz import plot_ngram_comparison
from visualizers.bias_viz import plot_bias_comparison, plot_political_spectrum
from visualizers.bow_viz import plot_word_frequencies, plot_word_comparisons
from visualizers.diff_viz import create_diff_heatmap, highlight_text_differences

def create_visualization_screen():
    """
    Create the visualization screen interface
    
    Returns:
        tuple: (viz_options, viz_params, viz_output)
    """
    with gr.Column() as viz_screen:
        gr.Markdown("## Visualization")
        gr.Markdown("Select visualization type and parameters to explore the analysis results.")
        
        # Main visualization selection
        with gr.Row():
            viz_options = gr.Dropdown(
                choices=[
                    "Topic Distribution",
                    "Topic Word Cloud",
                    "N-gram Comparison",
                    "Bias Comparison",
                    "Political Spectrum",
                    "Word Frequencies",
                    "Word Comparisons",
                    "Difference Heatmap",
                    "Text Differences"
                ],
                value="Topic Distribution",
                label="Visualization Type"
            )
            
            prompt_selector = gr.Dropdown(
                choices=[],  # Will be populated based on analysis results
                label="Select Prompt",
                interactive=True
            )
        
        # Visualization parameters
        with gr.Accordion("Visualization Parameters", open=True) as viz_params:
            # Parameters that change based on visualization type
            with gr.Group(visible=True) as topic_params:
                topic_num = gr.Slider(minimum=1, maximum=5, value=1, step=1, 
                                     label="Topic Number to Visualize")
            
            with gr.Group(visible=False) as ngram_params:
                ngram_models = gr.CheckboxGroup(
                    choices=[],  # Will be populated based on dataset
                    label="Models to Compare"
                )
                
            with gr.Group(visible=False) as bias_params:
                bias_type = gr.Radio(
                    choices=["Sentiment", "Partisan Leaning", "Framing"],
                    value="Partisan Leaning",
                    label="Bias Type to Visualize"
                )
                
            with gr.Group(visible=False) as word_params:
                word_count = gr.Slider(minimum=5, maximum=50, value=20, step=5,
                                      label="Number of Words to Display")
                
            with gr.Group(visible=False) as diff_params:
                model_pair = gr.Dropdown(
                    choices=[],  # Will be populated based on dataset
                    label="Model Pair to Compare"
                )
        
        # Visualization output area (supports multiple output types)
        with gr.Group() as viz_output:
            image_output = gr.Image(label="Visualization", visible=True)
            text_output = gr.HTML(label="Text Visualization", visible=False)
            
        # Update parameters based on selected visualization type
        viz_options.change(
            fn=update_viz_params,
            inputs=[viz_options],
            outputs=[
                topic_params, ngram_params, bias_params, 
                word_params, diff_params, image_output, text_output
            ]
        )
    
    return viz_options, viz_params, viz_output

def update_viz_params(viz_type):
    """
    Update visible parameters based on selected visualization type
    
    Args:
        viz_type (str): Selected visualization type
        
    Returns:
        tuple: Updated visibility for parameter groups and output components
    """
    # Default all parameter groups to invisible
    topic_visible = False
    ngram_visible = False
    bias_visible = False
    word_visible = False
    diff_visible = False
    
    # Default output types
    image_visible = True
    text_visible = False
    
    # Set visibility based on selected type
    if viz_type in ["Topic Distribution", "Topic Word Cloud"]:
        topic_visible = True
    elif viz_type == "N-gram Comparison":
        ngram_visible = True
    elif viz_type in ["Bias Comparison", "Political Spectrum"]:
        bias_visible = True
    elif viz_type in ["Word Frequencies", "Word Comparisons"]:
        word_visible = True
    elif viz_type in ["Difference Heatmap", "Text Differences"]:
        diff_visible = True
        if viz_type == "Text Differences":
            image_visible = False
            text_visible = True
    
    return (
        gr.update(visible=topic_visible),
        gr.update(visible=ngram_visible),
        gr.update(visible=bias_visible),
        gr.update(visible=word_visible),
        gr.update(visible=diff_visible),
        gr.update(visible=image_visible),
        gr.update(visible=text_visible)
    )

def update_visualization(analysis_results, viz_type, parameters):
    """
    Update the visualization based on selected type and parameters
    
    Args:
        analysis_results (dict): Results from analysis
        viz_type (str): Selected visualization type
        parameters (dict): Visualization parameters
        
    Returns:
        dict: Updates for visualization outputs
    """
    if not analysis_results or "analyses" not in analysis_results:
        # Return empty visualization if no results
        return gr.update(value=None, visible=True), gr.update(value="<p>No analysis results available</p>", visible=False)
    
    # Extract visualization parameters
    prompt = parameters.get("prompt")
    if not prompt or prompt not in analysis_results["analyses"]:
        # Use the first prompt if none selected or invalid
        prompt = list(analysis_results["analyses"].keys())[0]
    
    prompt_results = analysis_results["analyses"][prompt]
    
    # Generate visualization based on type
    try:
        if viz_type == "Topic Distribution":
            topic_num = parameters.get("topic_num", 1)
            if "topic_modeling" in prompt_results:
                fig = plot_topic_distribution(prompt_results["topic_modeling"], topic_num)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Topic Word Cloud":
            topic_num = parameters.get("topic_num", 1)
            if "topic_modeling" in prompt_results:
                fig = create_topic_wordcloud(prompt_results["topic_modeling"], topic_num)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "N-gram Comparison":
            selected_models = parameters.get("ngram_models")
            if "ngram_analysis" in prompt_results:
                fig = plot_ngram_comparison(prompt_results["ngram_analysis"], selected_models)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Bias Comparison":
            bias_type = parameters.get("bias_type", "Partisan Leaning")
            if "bias_detection" in prompt_results:
                fig = plot_bias_comparison(prompt_results["bias_detection"], bias_type)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Political Spectrum":
            if "bias_detection" in prompt_results:
                fig = plot_political_spectrum(prompt_results["bias_detection"])
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Word Frequencies":
            word_count = parameters.get("word_count", 20)
            if "bag_of_words" in prompt_results:
                fig = plot_word_frequencies(prompt_results["bag_of_words"], word_count)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Word Comparisons":
            word_count = parameters.get("word_count", 20)
            if "bag_of_words" in prompt_results:
                fig = plot_word_comparisons(prompt_results["bag_of_words"], word_count)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Difference Heatmap":
            model_pair = parameters.get("model_pair")
            if "difference_highlighting" in prompt_results:
                fig = create_diff_heatmap(prompt_results["difference_highlighting"], model_pair)
                return convert_fig_to_output(fig), gr.update(visible=False)
                
        elif viz_type == "Text Differences":
            model_pair = parameters.get("model_pair")
            if "difference_highlighting" in prompt_results:
                html = highlight_text_differences(prompt_results["difference_highlighting"], model_pair)
                return gr.update(visible=False), gr.update(value=html, visible=True)
                
        # Default fallback
        return gr.update(value=None, visible=True), gr.update(value="<p>No visualization available for the selected type</p>", visible=False)
        
    except Exception as e:
        error_message = f"<p>Error generating visualization: {str(e)}</p>"
        return gr.update(visible=False), gr.update(value=error_message, visible=True)

def convert_fig_to_output(fig):
    """
    Convert matplotlib figure to format suitable for Gradio Image output
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        numpy.ndarray: Image array for Gradio
    """
    # Save figure to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to image
    img_data = plt.imread(buf)
    plt.close(fig)  # Close the figure to free memory
    
    return img_data
import gradio as gr
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from difflib import SequenceMatcher

from visualization.ngram_visualizer import create_ngram_visualization
from visualization.topic_visualizer import process_and_visualize_topic_analysis  # Added import

def create_bow_visualization(analysis_results):
    """
    Create visualizations for bag of words analysis results
    
    Args:
        analysis_results (dict): Analysis results from the bow analysis
        
    Returns:
        list: List of gradio components with visualizations
    """
    # Parse analysis results if it's a string
    if isinstance(analysis_results, str):
        try:
            results = json.loads(analysis_results)
        except json.JSONDecodeError:
            return [gr.Markdown("Error parsing analysis results.")]
    else:
        results = analysis_results
    
    output_components = []
    
    # Check if we have valid results
    if not results or "analyses" not in results:
        return [gr.Markdown("No analysis results found.")]
    
    # Process each prompt
    for prompt, analyses in results["analyses"].items():
        output_components.append(gr.Markdown(f"## Analysis of Prompt: \"{prompt}\""))
        
        # Process Bag of Words analysis if available
        if "bag_of_words" in analyses:
            bow_results = analyses["bag_of_words"]
            
            # Show models being compared
            models = bow_results.get("models", [])
            if len(models) >= 2:
                output_components.append(gr.Markdown(f"### Comparing responses from {models[0]} and {models[1]}"))
                
                # Get important words for each model
                important_words = bow_results.get("important_words", {})
                
                # Prepare data for plotting important words
                if important_words:
                    for model_name, words in important_words.items():
                        df = pd.DataFrame(words)
                        
                        # Create bar chart for top words
                        fig = px.bar(df, x='word', y='count', 
                                     title=f"Top Words Used by {model_name}",
                                     labels={'word': 'Word', 'count': 'Frequency'},
                                     height=400)
                        
                        # Improve layout
                        fig.update_layout(
                            xaxis_title="Word",
                            yaxis_title="Frequency",
                            xaxis={'categoryorder':'total descending'}
                        )
                        
                        output_components.append(gr.Plot(value=fig))
                
                # Visualize differential words (words with biggest frequency difference)
                diff_words = bow_results.get("differential_words", [])
                word_matrix = bow_results.get("word_count_matrix", {})
                
                if diff_words and word_matrix and len(diff_words) > 0:
                    output_components.append(gr.Markdown("### Words with Biggest Frequency Differences"))
                    
                    # Create dataframe for plotting
                    model1, model2 = models[0], models[1]
                    diff_data = []
                    
                    for word in diff_words[:15]:  # Limit to top 15 for readability
                        if word in word_matrix:
                            counts = word_matrix[word]
                            diff_data.append({
                                "word": word,
                                model1: counts.get(model1, 0),
                                model2: counts.get(model2, 0)
                            })
                    
                    if diff_data:
                        diff_df = pd.DataFrame(diff_data)
                        
                        # Create grouped bar chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=diff_df['word'],
                            y=diff_df[model1],
                            name=model1,
                            marker_color='indianred'
                        ))
                        fig.add_trace(go.Bar(
                            x=diff_df['word'],
                            y=diff_df[model2],
                            name=model2,
                            marker_color='lightsalmon'
                        ))
                        
                        fig.update_layout(
                            title="Word Frequency Comparison",
                            xaxis_title="Word",
                            yaxis_title="Frequency",
                            barmode='group',
                            height=500
                        )
                        
                        output_components.append(gr.Plot(value=fig))
    
    # If no components were added, show a message
    if len(output_components) <= 1:
        output_components.append(gr.Markdown("No detailed Bag of Words analysis found in results."))
    
    return output_components


# update the process_and_visualize_analysis function
def process_and_visualize_analysis(analysis_results):
    """
    Process the analysis results and create visualization components

    Args:
        analysis_results (dict): The analysis results

    Returns:
        list: List of gradio components for visualization
    """
    try:
        print(f"Starting visualization of analysis results: {type(analysis_results)}")
        components = []

        if not analysis_results or "analyses" not in analysis_results:
            print("Warning: Empty or invalid analysis results")
            components.append(gr.Markdown("No analysis results to visualize."))
            return components

        # For each prompt in the analysis results
        for prompt, analyses in analysis_results.get("analyses", {}).items():
            print(f"Visualizing results for prompt: {prompt[:30]}...")
            components.append(gr.Markdown(f"## Analysis for Prompt:\n\"{prompt}\""))

            # Check for Bag of Words analysis
            if "bag_of_words" in analyses:
                print("Processing Bag of Words visualization")
                components.append(gr.Markdown("### Bag of Words Analysis"))
                bow_results = analyses["bag_of_words"]

                # Display models compared
                if "models" in bow_results:
                    models = bow_results["models"]
                    components.append(gr.Markdown(f"**Models compared**: {', '.join(models)}"))

                # Display important words for each model
                if "important_words" in bow_results:
                    components.append(gr.Markdown("#### Most Common Words by Model"))

                    for model, words in bow_results["important_words"].items():
                        print(f"Creating word list for model {model}")
                        word_list = [f"{item['word']} ({item['count']})" for item in words[:10]]
                        components.append(gr.Markdown(f"**{model}**: {', '.join(word_list)}"))

                # Add visualizations for word frequency differences
                if "differential_words" in bow_results and "word_count_matrix" in bow_results and len(
                        bow_results["models"]) >= 2:
                    diff_words = bow_results["differential_words"]
                    word_matrix = bow_results["word_count_matrix"]
                    models = bow_results["models"]

                    if diff_words and word_matrix and len(diff_words) > 0:
                        components.append(gr.Markdown("### Words with Biggest Frequency Differences"))

                        # Create dataframe for plotting
                        model1, model2 = models[0], models[1]
                        diff_data = []

                        for word in diff_words[:10]:  # Limit to top 10 for readability
                            if word in word_matrix:
                                counts = word_matrix[word]
                                model1_count = counts.get(model1, 0)
                                model2_count = counts.get(model2, 0)

                                # Only include if there's a meaningful difference
                                if abs(model1_count - model2_count) > 0:
                                    components.append(gr.Markdown(
                                        f"- **{word}**: {model1}: {model1_count}, {model2}: {model2_count}"
                                    ))

            # Check for N-gram analysis
            if "ngram_analysis" in analyses:
                print("Processing N-gram visualization")
                # Use the dedicated n-gram visualization function
                ngram_components = create_ngram_visualization(
                    {"analyses": {prompt: {"ngram_analysis": analyses["ngram_analysis"]}}})
                components.extend(ngram_components)
            
            # Check for Topic Modeling analysis
            if "topic_modeling" in analyses:
                print("Processing Topic Modeling visualization")
                # Use the dedicated topic visualization function
                topic_components = process_and_visualize_topic_analysis(
                    {"analyses": {prompt: {"topic_modeling": analyses["topic_modeling"]}}})
                components.extend(topic_components)

        if not components:
            components.append(gr.Markdown("No visualization components could be created from the analysis results."))

        print(f"Visualization complete: generated {len(components)} components")
        return components
    except Exception as e:
        import traceback
        error_msg = f"Visualization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [gr.Markdown(f"**Error during visualization:**\n\n```\n{error_msg}\n```")]


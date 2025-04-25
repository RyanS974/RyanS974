"""
Visualization components for RoBERTa sentiment analysis
"""
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

def create_sentiment_visualization(analysis_results):
    """
    Create visualizations for RoBERTa sentiment analysis results
    
    Args:
        analysis_results (dict): Analysis results from the sentiment analysis
        
    Returns:
        list: List of gradio components with visualizations
    """
    print("Starting create_sentiment_visualization function")
    output_components = []
    
    # Check if we have valid results
    if not analysis_results or "analyses" not in analysis_results:
        print("No analysis results found.")
        return [gr.Markdown("No analysis results found.")]
    
    # Add debug print for each step
    print(f"Number of prompts: {len(analysis_results['analyses'])}")
    
    # Process each prompt
    for prompt, analyses in analysis_results["analyses"].items():
        output_components.append(gr.Markdown(f"## Analysis of Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\""))
        
        # Process RoBERTa sentiment analysis if available
        if "roberta_sentiment" in analyses:
            sentiment_results = analyses["roberta_sentiment"]
            
            # Check if there's an error
            if "error" in sentiment_results:
                output_components.append(gr.Markdown(f"**Error in sentiment analysis:** {sentiment_results['error']}"))
                continue
            
            # Show models being compared
            models = sentiment_results.get("models", [])
            if len(models) >= 2:
                output_components.append(gr.Markdown(f"### RoBERTa Sentiment Analysis: Comparing {models[0]} and {models[1]}"))
                
                # Create text-based summary of sentiment scores
                sa_data = sentiment_results.get("sentiment_analysis", {})
                if sa_data and len(models) >= 2:
                    # Extract sentiment scores and labels for comparison 
                    model_data = []
                    
                    summary_html = "<div style='margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>"
                    summary_html += "<h4 style='margin-top: 0;'>Sentiment Score Comparison</h4>"
                    summary_html += "<table style='width: 100%; border-collapse: collapse;'>"
                    summary_html += "<tr><th style='text-align: left; padding: 8px; border-bottom: 1px solid #ddd;'>Model</th>"
                    summary_html += "<th style='text-align: center; padding: 8px; border-bottom: 1px solid #ddd;'>Sentiment Score</th>"
                    summary_html += "<th style='text-align: center; padding: 8px; border-bottom: 1px solid #ddd;'>Label</th></tr>"
                    
                    for model_name in models:
                        if model_name in sa_data:
                            model_result = sa_data.get(model_name)
                            if model_result is not None:
                                score = model_result.get("sentiment_score", 0)
                                label = model_result.get("label", "neutral").capitalize()
                            else:
                                score = 0
                                label = "Neutral"
                            
                            # Set color based on sentiment
                            if label.lower() == "positive":
                                color = "green"
                            elif label.lower() == "negative":
                                color = "red"
                            else:
                                color = "gray"
                                
                            summary_html += f"<tr>"
                            summary_html += f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>{model_name}</td>"
                            summary_html += f"<td style='text-align: center; padding: 8px; border-bottom: 1px solid #ddd;'>{score:.2f}</td>"
                            summary_html += f"<td style='text-align: center; padding: 8px; border-bottom: 1px solid #ddd; color: {color}; font-weight: bold;'>{label}</td>"
                            summary_html += f"</tr>"
                    
                    summary_html += "</table></div>"
                    output_components.append(gr.HTML(summary_html))
                    
                    # Create HTML-based score comparison gauge
                    model_scores = []
                    for model_name in models:
                        if model_name in sa_data:
                            model_result = sa_data.get(model_name)
                            if model_result is not None:
                                score = model_result.get("sentiment_score", 0)
                                model_scores.append((model_name, score))
                    
                    if len(model_scores) >= 2:
                        gauge_html = "<div style='margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>"
                        gauge_html += "<h4 style='text-align: center; margin-top: 0;'>Sentiment Scale</h4>"
                        gauge_html += "<div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>"
                        gauge_html += "<span>Very Negative (-2.0)</span>"
                        gauge_html += "<span>Neutral (0.0)</span>"
                        gauge_html += "<span>Very Positive (2.0)</span>"
                        gauge_html += "</div>"
                        
                        # Create the gauge background
                        gauge_html += "<div style='position: relative; width: 100%; height: 30px; background: linear-gradient(to right, #d73027, #f46d43, #fdae61, #fee08b, #ffffbf, #d9ef8b, #a6d96a, #66bd63, #1a9850); border-radius: 5px;'>"
                        
                        # Add model markers
                        for model_name, score in model_scores:
                            # Calculate position (0-100%)
                            position = ((score + 2.0) / 4.0) * 100
                            position = max(0, min(100, position))  # Clamp between 0-100%
                            
                            # Calculate color
                            if score > 0.5:
                                color = "#006400"  # Dark green
                            elif score < -0.5:
                                color = "#8B0000"  # Dark red
                            else:
                                color = "#000000"  # Black
                            
                            gauge_html += f"<div style='position: absolute; left: {position}%; transform: translateX(-50%); top: 0;'>"
                            gauge_html += f"<div style='width: 3px; height: 30px; background-color: {color};'></div>"
                            gauge_html += f"<div style='position: absolute; top: 100%; left: 50%; transform: translateX(-50%); white-space: nowrap; font-weight: bold; color: {color};'>{model_name}: {score:.2f}</div>"
                            gauge_html += "</div>"
                        
                        gauge_html += "</div></div>"
                        
                        output_components.append(gr.HTML(gauge_html))
                
                # Display comparison summary
                if "comparison" in sentiment_results:
                    comparison = sentiment_results["comparison"]
                    
                    summary_html = """
                    <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                        <h4 style="margin-top: 0;">Sentiment Comparison Summary</h4>
                    """
                    
                    # Add difference direction
                    if "difference_direction" in comparison:
                        summary_html += f"""
                        <p style="font-weight: 500; margin-bottom: 10px;">
                            {comparison["difference_direction"]}
                        </p>
                        """
                    
                    # Add significance info
                    if "significant_difference" in comparison:
                        color = "red" if comparison["significant_difference"] else "green"
                        significance = "Significant" if comparison["significant_difference"] else "Minor"
                        
                        summary_html += f"""
                        <p>
                            <span style="font-weight: bold; color: {color};">{significance} difference</span> in sentiment 
                            (difference score: {comparison.get("sentiment_difference", 0):.2f})
                        </p>
                        """
                    
                    summary_html += "</div>"
                    output_components.append(gr.HTML(summary_html))
                
                # Display sentence-level sentiment analysis for both responses
                model_sentences = {}
                
                for model_name in models:
                    if model_name in sa_data:
                        model_result = sa_data.get(model_name)
                        if model_result is not None and "sentence_scores" in model_result:
                            sentence_scores = model_result.get("sentence_scores")
                            if sentence_scores:
                                model_sentences[model_name] = sentence_scores
                
                if model_sentences and any(len(sentences) > 0 for sentences in model_sentences.values()):
                    output_components.append(gr.Markdown("### Sentence-Level Sentiment Analysis"))
                    
                    for model_name, sentences in model_sentences.items():
                        if sentences:
                            output_components.append(gr.Markdown(f"#### {model_name} Response Breakdown"))
                            
                            # Create HTML visualization for sentences with sentiment
                            sentences_html = """
                            <div style="margin-bottom: 20px;">
                            """
                            
                            for i, sentence in enumerate(sentences):
                                score = sentence.get("score", 0)
                                label = sentence.get("label", "neutral")
                                text = sentence.get("text", "")
                                
                                # Skip very short sentences or empty text
                                if len(text.split()) < 3:
                                    continue
                                
                                # Color based on sentiment
                                if label == "positive":
                                    color = f"rgba(0, 128, 0, {min(1.0, abs(score) * 0.5)})"
                                    border = "rgba(0, 128, 0, 0.3)"
                                elif label == "negative":
                                    color = f"rgba(255, 0, 0, {min(1.0, abs(score) * 0.5)})"
                                    border = "rgba(255, 0, 0, 0.3)"
                                else:
                                    color = "rgba(128, 128, 128, 0.1)"
                                    border = "rgba(128, 128, 128, 0.3)"
                                
                                sentences_html += f"""
                                <div style="padding: 10px; margin-bottom: 10px; background-color: {color}; 
                                            border-radius: 5px; border: 1px solid {border};">
                                    <div style="display: flex; justify-content: space-between;">
                                        <span>{text}</span>
                                        <span style="margin-left: 10px; font-weight: bold;">
                                            {score:.2f} ({label.capitalize()})
                                        </span>
                                    </div>
                                </div>
                                """
                            
                            sentences_html += "</div>"
                            output_components.append(gr.HTML(sentences_html))
    
    # If no components were added, show a message
    if len(output_components) <= 1:
        output_components.append(gr.Markdown("No detailed sentiment analysis found in results."))
    
    return output_components

def process_and_visualize_sentiment_analysis(analysis_results):
    """
    Process the sentiment analysis results and create visualization components
    
    Args:
        analysis_results (dict): The analysis results
        
    Returns:
        list: List of gradio components for visualization
    """
    try:
        print(f"Starting visualization of sentiment analysis results")
        components = create_sentiment_visualization(analysis_results)
        return components
    except Exception as e:
        import traceback
        error_msg = f"Sentiment visualization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [
            gr.Markdown(f"**Error during sentiment visualization:**"),
            gr.HTML(f"<div style='background-color: #FEE; padding: 10px; border-radius: 5px; border: 1px solid #F88;'>" +
                   f"<pre style='white-space: pre-wrap; overflow-wrap: break-word;'>{str(e)}</pre></div>")
        ]
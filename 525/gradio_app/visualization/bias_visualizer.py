import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_bias_visualization(analysis_results):
    """
    Create visualizations for bias detection analysis results
    
    Args:
        analysis_results (dict): Analysis results from the bias detection
        
    Returns:
        list: List of gradio components with visualizations
    """
    output_components = []
    
    # Check if we have valid results
    if not analysis_results or "analyses" not in analysis_results:
        return [gr.Markdown("No analysis results found.")]
    
    # Process each prompt
    for prompt, analyses in analysis_results["analyses"].items():
        # Process Bias Detection analysis if available
        if "bias_detection" in analyses:
            bias_results = analyses["bias_detection"]
            
            # Show models being compared
            models = bias_results.get("models", [])
            if len(models) >= 2:
                output_components.append(gr.Markdown(f"### Bias Analysis: Comparing responses from {models[0]} and {models[1]}"))
                
                # Check if there's an error
                if "error" in bias_results:
                    output_components.append(gr.Markdown(f"**Error in bias detection:** {bias_results['error']}"))
                    continue
                
                model1_name, model2_name = models[0], models[1]
                
                # Comparative results
                if "comparative" in bias_results:
                    comparative = bias_results["comparative"]
                    
                    output_components.append(gr.Markdown("#### Comparative Bias Analysis"))
                    
                    # Create summary table
                    summary_html = f"""
                    <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Bias Category</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">{model1_name}</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">{model2_name}</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Significant Difference?</th>
                    </tr>
                    """
                    
                    # Sentiment row
                    if "sentiment" in comparative:
                        sent_sig = comparative["sentiment"].get("significant", False)
                        summary_html += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Sentiment Bias</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["sentiment"].get(model1_name, "N/A").title()}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["sentiment"].get(model2_name, "N/A").title()}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; color: {'red' if sent_sig else 'green'}">{"Yes" if sent_sig else "No"}</td>
                        </tr>
                        """
                    
                    # Partisan row
                    if "partisan" in comparative:
                        part_sig = comparative["partisan"].get("significant", False)
                        summary_html += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Partisan Leaning</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["partisan"].get(model1_name, "N/A").title()}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["partisan"].get(model2_name, "N/A").title()}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; color: {'red' if part_sig else 'green'}">{"Yes" if part_sig else "No"}</td>
                        </tr>
                        """
                    
                    # Framing row
                    if "framing" in comparative:
                        frame_diff = comparative["framing"].get("different_frames", False)
                        summary_html += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Dominant Frame</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["framing"].get(model1_name, "N/A").title().replace('_', ' ')}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{comparative["framing"].get(model2_name, "N/A").title().replace('_', ' ')}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; color: {'red' if frame_diff else 'green'}">{"Yes" if frame_diff else "No"}</td>
                        </tr>
                        """
                    
                    # Overall row
                    if "overall" in comparative:
                        overall_sig = comparative["overall"].get("significant_bias_difference", False)
                        summary_html += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Overall Bias Difference</td>
                            <td colspan="2" style="border: 1px solid #ddd; padding: 8px; text-align: center;">{comparative["overall"].get("difference", 0):.2f} / 1.0</td>
                            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; color: {'red' if overall_sig else 'green'}">{"Yes" if overall_sig else "No"}</td>
                        </tr>
                        """
                    
                    summary_html += "</table>"
                    
                    # Add the HTML table to the components
                    output_components.append(gr.HTML(summary_html))
                
                # Create detailed visualizations for each model if available
                for model_name in [model1_name, model2_name]:
                    if model_name in bias_results:
                        model_data = bias_results[model_name]
                        
                        # Sentiment visualization
                        if "sentiment" in model_data:
                            sentiment = model_data["sentiment"]
                            if "sentiment_scores" in sentiment:
                                # Create sentiment score chart
                                sentiment_df = pd.DataFrame({
                                    'Score': [
                                        sentiment["sentiment_scores"]["pos"],
                                        sentiment["sentiment_scores"]["neg"],
                                        sentiment["sentiment_scores"]["neu"]
                                    ],
                                    'Category': ['Positive', 'Negative', 'Neutral']
                                })
                                
                                fig = px.bar(
                                    sentiment_df,
                                    x='Category',
                                    y='Score',
                                    title=f"Sentiment Analysis for {model_name}",
                                    height=300
                                )
                                
                                output_components.append(gr.Plot(value=fig))
                        
                        # Partisan leaning visualization
                        if "partisan" in model_data:
                            partisan = model_data["partisan"]
                            if "liberal_count" in partisan and "conservative_count" in partisan:
                                # Create partisan terms chart
                                partisan_df = pd.DataFrame({
                                    'Count': [partisan["liberal_count"], partisan["conservative_count"]],
                                    'Category': ['Liberal Terms', 'Conservative Terms']
                                })
                                
                                fig = px.bar(
                                    partisan_df,
                                    x='Category',
                                    y='Count',
                                    title=f"Partisan Term Usage for {model_name}",
                                    color='Category',
                                    color_discrete_map={
                                        'Liberal Terms': 'blue',
                                        'Conservative Terms': 'red'
                                    },
                                    height=300
                                )
                                
                                output_components.append(gr.Plot(value=fig))
                            
                            # Show example partisan terms
                            if "liberal_terms" in partisan or "conservative_terms" in partisan:
                                lib_terms = ", ".join(partisan.get("liberal_terms", []))
                                con_terms = ", ".join(partisan.get("conservative_terms", []))
                                
                                if lib_terms or con_terms:
                                    terms_md = f"**Partisan Terms Used by {model_name}**\n\n"
                                    if lib_terms:
                                        terms_md += f"- Liberal terms: {lib_terms}\n"
                                    if con_terms:
                                        terms_md += f"- Conservative terms: {con_terms}\n"
                                    
                                    output_components.append(gr.Markdown(terms_md))
                        
                        # Framing visualization
                        if "framing" in model_data:
                            framing = model_data["framing"]
                            if "framing_distribution" in framing:
                                # Create framing distribution chart
                                frame_items = []
                                for frame, value in framing["framing_distribution"].items():
                                    frame_items.append({
                                        'Frame': frame.replace('_', ' ').title(),
                                        'Proportion': value
                                    })
                                
                                frame_df = pd.DataFrame(frame_items)
                                
                                fig = px.pie(
                                    frame_df,
                                    values='Proportion',
                                    names='Frame',
                                    title=f"Issue Framing Distribution for {model_name}",
                                    height=400
                                )
                                
                                output_components.append(gr.Plot(value=fig))
                            
                            # Show example framing terms
                            if "framing_examples" in framing:
                                examples_md = f"**Example Framing Terms Used by {model_name}**\n\n"
                                for frame, examples in framing["framing_examples"].items():
                                    if examples:
                                        examples_md += f"- {frame.replace('_', ' ').title()}: {', '.join(examples)}\n"
                                
                                output_components.append(gr.Markdown(examples_md))
    
    # If no components were added, show a message
    if len(output_components) <= 1:
        output_components.append(gr.Markdown("No detailed bias detection analysis found in results."))
    
    return output_components

def process_and_visualize_bias_analysis(analysis_results):
    """
    Process the bias detection analysis results and create visualization components
    
    Args:
        analysis_results (dict): The analysis results
        
    Returns:
        list: List of gradio components for visualization
    """
    try:
        print(f"Starting visualization of bias detection analysis results")
        return create_bias_visualization(analysis_results)
    except Exception as e:
        import traceback
        error_msg = f"Bias detection visualization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [gr.Markdown(f"**Error during bias detection visualization:**\n\n```\n{error_msg}\n```")]
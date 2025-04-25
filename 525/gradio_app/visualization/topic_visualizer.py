"""
Enhanced visualization for topic modeling analysis results
"""
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_topic_visualization(analysis_results):
    """
    Create enhanced visualizations for topic modeling analysis results
    
    Args:
        analysis_results (dict): Analysis results from the topic modeling analysis
        
    Returns:
        list: List of gradio components with visualizations
    """
    # Initialize output components list
    output_components = []
    
    # Check if we have valid results
    if not analysis_results or "analyses" not in analysis_results:
        return [gr.Markdown("No analysis results found.")]
    
    # Process each prompt
    for prompt, analyses in analysis_results["analyses"].items():
        # Process Topic Modeling analysis if available
        if "topic_modeling" in analyses:
            topic_results = analyses["topic_modeling"]
            
            # Enhanced error checking and messaging
            if "error" in topic_results:
                output_components.append(gr.Markdown(f"## ⚠️ Topic Modeling Error"))
                output_components.append(gr.Markdown(f"Error: {topic_results['error']}"))
                output_components.append(gr.Markdown("Suggestions:"))
                output_components.append(gr.Markdown("1. Try with longer text samples - topic modeling typically needs 100+ words per document"))
                output_components.append(gr.Markdown("2. Reduce the number of topics (2-3 for short texts)"))
                output_components.append(gr.Markdown("3. Try the Bag of Words or N-gram analysis for shorter texts"))
                continue
            
            # Show method and number of topics
            method = topic_results.get("method", "lda").upper()
            n_topics = topic_results.get("n_topics", 3)
            
            # Check if n_topics was adjusted
            if "adjusted_n_topics" in topic_results and topic_results["adjusted_n_topics"] != topic_results.get("original_n_topics", n_topics):
                output_components.append(gr.Markdown(
                    f"## Topic Modeling Analysis ({method}, {topic_results['adjusted_n_topics']} topics) " +
                    f"*Adjusted from {topic_results['original_n_topics']} due to limited text content*"
                ))
                n_topics = topic_results["adjusted_n_topics"]
            else:
                output_components.append(gr.Markdown(f"## Topic Modeling Analysis ({method}, {n_topics} topics)"))
            
            # Check for warnings
            if "warnings" in topic_results:
                if isinstance(topic_results["warnings"], list):
                    for warning in topic_results["warnings"]:
                        output_components.append(gr.Markdown(f"⚠️ **Warning**: {warning}"))
                else:
                    output_components.append(gr.Markdown(f"⚠️ **Warning**: {topic_results['warnings']}"))
            
            if "warning" in topic_results:
                output_components.append(gr.Markdown(f"⚠️ **Warning**: {topic_results['warning']}"))
            
            # Show models being compared
            models = topic_results.get("models", [])
            if len(models) >= 2:
                output_components.append(gr.Markdown(f"### Comparing responses from {models[0]} and {models[1]}"))
                
                # Show topic quality metrics if available
                if "coherence_scores" in topic_results:
                    coherence_html = f"""
                    <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                    <h4 style="margin-top: 0;">Topic Quality Metrics</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Metric</th>
                        <th style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">{models[0]}</th>
                        <th style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">{models[1]}</th>
                        <th style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">Combined</th>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">Topic Coherence</td>
                        <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">
                            {topic_results["coherence_scores"].get(models[0], 0):.2f}
                        </td>
                        <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">
                            {topic_results["coherence_scores"].get(models[1], 0):.2f}
                        </td>
                        <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">
                            {topic_results["coherence_scores"].get("combined", 0):.2f}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">Topic Diversity</td>
                        <td style="text-align: center; padding: 8px;">
                            {topic_results["diversity_scores"].get(models[0], 0):.2f}
                        </td>
                        <td style="text-align: center; padding: 8px;">
                            {topic_results["diversity_scores"].get(models[1], 0):.2f}
                        </td>
                        <td style="text-align: center; padding: 8px;">
                            {topic_results["diversity_scores"].get("combined", 0):.2f}
                        </td>
                    </tr>
                    </table>
                    <p style="margin-bottom: 0; font-size: 0.9em; color: #666;">
                        Higher coherence scores indicate more semantically coherent topics.<br>
                        Higher diversity scores indicate less overlap between topics.
                    </p>
                    </div>
                    """
                    output_components.append(gr.HTML(coherence_html))
                
                # Visualize topics
                topics = topic_results.get("topics", [])
                if topics:
                    output_components.append(gr.Markdown("### Discovered Topics"))
                    
                    # Create a topic word cloud using HTML/CSS for better visibility
                    for topic in topics:
                        topic_id = topic.get("id", 0)
                        words = topic.get("words", [])
                        weights = topic.get("weights", [])
                        
                        if words and weights and len(words) == len(weights):
                            # Generate a word cloud-like div using HTML/CSS
                            word_cloud_html = f"""
                            <div style="margin-bottom: 25px;">
                                <h4 style="margin-bottom: 10px;">Topic {topic_id+1}</h4>
                                <div style="display: flex; flex-wrap: wrap; gap: 10px; background: #f9f9f9; padding: 15px; border-radius: 5px;">
                            """
                            
                            # Sort words by weight for better visualization
                            word_weight_pairs = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)
                            
                            # Add each word with size based on weight
                            for word, weight in word_weight_pairs:
                                # Scale weight to a reasonable font size (min 14px, max 28px)
                                font_size = 14 + min(14, round(weight * 30))
                                # Color based on weight (darker = higher weight)
                                color_intensity = max(0, min(90, int(100 - weight * 100)))
                                color = f"hsl(210, 70%, {color_intensity}%)"
                                
                                word_cloud_html += f"""
                                <span style="font-size: {font_size}px; color: {color}; margin: 3px; 
                                padding: 5px; border-radius: 3px; background: rgba(0,0,0,0.03);">
                                {word}
                                </span>
                                """
                            
                            word_cloud_html += """
                                </div>
                            </div>
                            """
                            
                            output_components.append(gr.HTML(word_cloud_html))
                    
                    # Add a proper bar chart visualization for topic words
                    for topic in topics[:min(3, len(topics))]:  # Show charts for max 3 topics to avoid clutter
                        topic_id = topic.get("id", 0)
                        words = topic.get("words", [])
                        weights = topic.get("weights", [])
                        
                        if words and weights and len(words) == len(weights):
                            # Create dataframe for plotting
                            df = pd.DataFrame({
                                'word': words,
                                'weight': weights
                            })
                            
                            # Sort by weight
                            df = df.sort_values('weight', ascending=False)
                            
                            # Limit to top N words for clarity
                            df = df.head(10)
                            
                            # Create bar chart
                            fig = px.bar(
                                df, x='weight', y='word',
                                title=f"Topic {topic_id+1} Top Words",
                                labels={'word': 'Word', 'weight': 'Weight'},
                                height=300,
                                orientation='h'  # Horizontal bars
                            )
                            
                            # Improve layout
                            fig.update_layout(
                                margin=dict(l=10, r=10, t=40, b=10),
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            
                            output_components.append(gr.Plot(value=fig))
                
                # Visualize topic distributions for each model
                model_topics = topic_results.get("model_topics", {})
                if model_topics and all(model in model_topics for model in models):
                    output_components.append(gr.Markdown("### Topic Distribution by Model"))
                    
                    # Create multi-model topic distribution comparison
                    distribution_data = []
                    for model in models:
                        if model in model_topics:
                            distribution = model_topics[model]
                            for i, weight in enumerate(distribution):
                                if i < 10:  # Limit to 10 topics max
                                    distribution_data.append({
                                        'Model': model,
                                        'Topic': f"Topic {i+1}",
                                        'Weight': weight
                                    })
                    
                    if distribution_data:
                        df = pd.DataFrame(distribution_data)
                        
                        # Create grouped bar chart
                        fig = px.bar(
                            df, x='Topic', y='Weight', color='Model',
                            barmode='group',
                            title="Topic Distribution Comparison",
                            height=400
                        )
                        
                        output_components.append(gr.Plot(value=fig))
                
                # Visualize topic differences as a heatmap
                comparisons = topic_results.get("comparisons", {})
                if comparisons:
                    comparison_key = f"{models[0]} vs {models[1]}"
                    if comparison_key in comparisons:
                        output_components.append(gr.Markdown("### Topic Similarity Analysis"))
                        
                        # Get JS divergence
                        js_divergence = comparisons[comparison_key].get("js_divergence", 0)
                        
                        # Create a divergence meter
                        divergence_html = f"""
                        <div style="margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">Topic Distribution Divergence</h4>
                            <div style="display: flex; align-items: center; justify-content: center;">
                                <div style="width: 300px; height: 40px; background: linear-gradient(to right, #1a9850, #ffffbf, #d73027); border-radius: 5px; position: relative; margin: 10px 0;">
                                    <div style="position: absolute; height: 40px; width: 2px; background-color: #000; left: {min(300, max(0, js_divergence * 300))}px;"></div>
                                </div>
                            </div>
                            <div style="display: flex; justify-content: space-between; width: 300px; margin: 0 auto;">
                                <span>Similar (0.0)</span>
                                <span>Different (1.0)</span>
                            </div>
                            <p style="margin-top: 10px; font-weight: bold;">Score: {js_divergence:.3f}</p>
                            <p style="margin-bottom: 0; font-size: 0.9em; color: #666;">
                                Jensen-Shannon Divergence measures the similarity between topic distributions.<br>
                                Lower values indicate more similar topic distributions between models.
                            </p>
                        </div>
                        """
                        
                        output_components.append(gr.HTML(divergence_html))
                        
                        # Create similarity matrix heatmap if available
                        similarity_matrix = topic_results.get("similarity_matrix", [])
                        if similarity_matrix and len(similarity_matrix) > 0:
                            # Convert to format for heatmap
                            z_data = similarity_matrix
                            
                            # Create heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=z_data,
                                x=[f"{models[1]} Topic {i+1}" for i in range(len(similarity_matrix[0]))],
                                y=[f"{models[0]} Topic {i+1}" for i in range(len(similarity_matrix))],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Similarity")
                            ))
                            
                            fig.update_layout(
                                title="Topic Similarity Matrix",
                                height=400,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )
                            
                            output_components.append(gr.Plot(value=fig))
                
                # Show best matching topics
                matched_topics = topic_results.get("matched_topics", [])
                if matched_topics:
                    output_components.append(gr.Markdown("### Most Similar Topic Pairs"))
                    
                    # Create HTML table for matched topics
                    matched_topics_html = """
                    <div style="margin: 20px 0;">
                    <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">Topic Pair</th>
                        <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">Top Words in Model 1</th>
                        <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">Top Words in Model 2</th>
                        <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: center;">Similarity</th>
                    </tr>
                    """
                    
                    # Sort by similarity, highest first
                    sorted_matches = sorted(matched_topics, key=lambda x: x['similarity'], reverse=True)
                    
                    for match in sorted_matches:
                        # Format words with commas
                        words1 = ", ".join(match["set1_topic_words"][:5])  # Show top 5 words
                        words2 = ", ".join(match["set2_topic_words"][:5])  # Show top 5 words
                        
                        # Calculate color based on similarity (green for high, red for low)
                        similarity = match["similarity"]
                        color = f"hsl({int(120 * similarity)}, 70%, 50%)"
                        
                        matched_topics_html += f"""
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                                {models[0]} Topic {match['set1_topic_id']+1} ↔ {models[1]} Topic {match['set2_topic_id']+1}
                            </td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{words1}</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{words2}</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: center; font-weight: bold; color: {color};">
                                {similarity:.2f}
                            </td>
                        </tr>
                        """
                    
                    matched_topics_html += """
                    </table>
                    </div>
                    """
                    
                    output_components.append(gr.HTML(matched_topics_html))
    
    # If no components were added, show a message
    if len(output_components) <= 1:
        output_components.append(gr.Markdown("No detailed Topic Modeling analysis found in results."))
    
    return output_components


def process_and_visualize_topic_analysis(analysis_results):
    """
    Process the topic modeling analysis results and create visualization components
    
    Args:
        analysis_results (dict): The analysis results
        
    Returns:
        list: List of gradio components for visualization
    """
    try:
        print(f"Starting visualization of topic modeling analysis results")
        components = create_topic_visualization(analysis_results)
        print(f"Completed topic modeling visualization with {len(components)} components")
        return components
    except Exception as e:
        import traceback
        error_msg = f"Topic modeling visualization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [
            gr.Markdown(f"**Error during topic modeling visualization:**"),
            gr.Markdown(f"```\n{str(e)}\n```"),
            gr.Markdown("Try adjusting the number of topics or using longer text inputs.")
        ]
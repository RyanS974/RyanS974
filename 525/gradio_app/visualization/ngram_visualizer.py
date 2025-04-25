import gradio as gr
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_ngram_visualization(analysis_results):
    """
    Create visualizations for n-gram analysis results

    Args:
        analysis_results (dict): Analysis results from the n-gram analysis

    Returns:
        list: List of gradio components with visualizations
    """
    output_components = []

    # Check if we have valid results
    if not analysis_results or "analyses" not in analysis_results:
        return [gr.Markdown("No analysis results found.")]

    # Process each prompt
    for prompt, analyses in analysis_results["analyses"].items():
        # Process N-gram analysis if available
        if "ngram_analysis" in analyses:
            ngram_results = analyses["ngram_analysis"]
            
            # Check if there's an error in the analysis
            if "error" in ngram_results:
                output_components.append(gr.Markdown(f"**Error in N-gram analysis:** {ngram_results['error']}"))
                continue

            # Show models being compared
            models = ngram_results.get("models", [])
            ngram_size = ngram_results.get("ngram_size", 2)
            size_name = "Unigrams" if ngram_size == 1 else f"{ngram_size}-grams"

            if len(models) >= 2:
                output_components.append(
                    gr.Markdown(f"### {size_name} Analysis: Comparing responses from {models[0]} and {models[1]}"))

                # Get important n-grams for each model
                important_ngrams = ngram_results.get("important_ngrams", {})

                # Display important n-grams for each model
                if important_ngrams:
                    for model_name, ngrams in important_ngrams.items():
                        output_components.append(gr.Markdown(f"#### Top {size_name} Used by {model_name}"))
                        
                        if ngrams:
                            # Create a formatted list of n-grams for display
                            ngram_list = [f"**{item['ngram']}** ({item['count']})" for item in ngrams[:10]]
                            output_components.append(gr.Markdown(", ".join(ngram_list)))
                        else:
                            output_components.append(gr.Markdown("No significant n-grams found."))

                        # Only if we have enough data, create a bar chart
                        if len(ngrams) >= 3:
                            try:
                                df = pd.DataFrame(ngrams)
                                # Create bar chart for top n-grams
                                fig = px.bar(df[:10], x='ngram', y='count',
                                             title=f"Top {size_name} Used by {model_name}",
                                             labels={'ngram': 'N-gram', 'count': 'Frequency'},
                                             height=400)

                                # Improve layout
                                fig.update_layout(
                                    xaxis_title="N-gram",
                                    yaxis_title="Frequency",
                                    xaxis={'categoryorder': 'total descending'}
                                )

                                output_components.append(gr.Plot(value=fig))
                            except Exception as e:
                                output_components.append(gr.Markdown(f"Visualization error: {str(e)}"))

                # Visualize differential n-grams (n-grams with biggest frequency difference)
                diff_ngrams = ngram_results.get("differential_ngrams", [])
                ngram_matrix = ngram_results.get("ngram_count_matrix", {})

                if diff_ngrams and ngram_matrix and len(diff_ngrams) > 0:
                    output_components.append(gr.Markdown(f"### {size_name} with Biggest Frequency Differences"))

                    # Create dataframe for plotting
                    model1, model2 = models[0], models[1]
                    diff_data = []

                    for ngram in diff_ngrams[:10]:  # Limit to top 10 for readability
                        if ngram in ngram_matrix:
                            counts = ngram_matrix[ngram]
                            model1_count = counts.get(model1, 0)
                            model2_count = counts.get(model2, 0)
                            
                            # Only include if there's a meaningful difference
                            if abs(model1_count - model2_count) > 0:
                                output_components.append(gr.Markdown(
                                    f"- **{ngram}**: {model1}: {model1_count}, {model2}: {model2_count}"
                                ))

                # Add similarity comparison if available
                if "comparisons" in ngram_results:
                    output_components.append(gr.Markdown("### N-gram Similarity Metrics"))
                    comparison_key = f"{models[0]} vs {models[1]}"

                    if comparison_key in ngram_results["comparisons"]:
                        metrics = ngram_results["comparisons"][comparison_key]
                        common_count = metrics.get("common_ngram_count", 0)

                        metrics_text = f"""
                        - **Common {size_name}**: {common_count} {size_name.lower()} appear in both responses
                        """

                        output_components.append(gr.Markdown(metrics_text))

    # If no components were added other than header, show a message
    if len(output_components) <= 1:
        output_components.append(gr.Markdown(f"No detailed N-gram analysis found in results."))

    return output_components


def process_and_visualize_ngram_analysis(analysis_results):
    """
    Process the n-gram analysis results and create visualization components

    Args:
        analysis_results (dict): The analysis results

    Returns:
        list: List of gradio components for visualization
    """
    try:
        print(f"Starting visualization of n-gram analysis results")
        return create_ngram_visualization(analysis_results)
    except Exception as e:
        import traceback
        error_msg = f"N-gram visualization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return [gr.Markdown(f"**Error during n-gram visualization:**\n\n```\n{error_msg}\n```")]

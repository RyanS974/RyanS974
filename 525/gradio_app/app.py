import gradio as gr
from ui.dataset_input import create_dataset_input, load_example_dataset
from ui.analysis_screen import create_analysis_screen, process_analysis_request
from ui.roberta_screen import create_roberta_screen, process_roberta_request
from visualization.bow_visualizer import process_and_visualize_analysis
from visualization.roberta_visualizer import process_and_visualize_sentiment_analysis
import nltk
import os
import json
import matplotlib.pyplot as plt
import io
import base64
import datetime
from PIL import Image

# Download necessary NLTK resources function remains unchanged
def download_nltk_resources():
    """Download required NLTK resources if not already downloaded""" 
    try:
        # Create nltk_data directory in the user's home directory if it doesn't exist
        nltk_data_path = os.path.expanduser("~/nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        
        # Add this path to NLTK's data path
        nltk.data.path.append(nltk_data_path)
        
        # Download required resources
        resources = ['punkt', 'wordnet', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                # Different resources can be in different directories in NLTK
                locations = [
                    f'tokenizers/{resource}',
                    f'corpora/{resource}',
                    f'taggers/{resource}',
                    f'{resource}'
                ]
                
                found = False
                for location in locations:
                    try:
                        nltk.data.find(location)
                        print(f"Resource {resource} already downloaded")
                        found = True
                        break
                    except LookupError:
                        continue
                
                if not found:
                    print(f"Downloading {resource}...")
                    nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error with resource {resource}: {e}")
        
        print("NLTK resources check completed")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

def create_app():
    """
    Create a streamlined Gradio app for dataset input and analysis. 
    
    Returns:
        gr.Blocks: The Gradio application
    """
    with gr.Blocks(title="LLM Response Comparator") as app:
        # Application state to share data between tabs
        dataset_state = gr.State({})
        analysis_results_state = gr.State({})
        roberta_results_state = gr.State({})
        
        # Add a state for storing user dataset analysis results
        user_analysis_log = gr.State({})
        
        # Dataset Input Tab
        with gr.Tab("Dataset Input"):
            # Filter out files that start with 'summary' for the Dataset Input tab
            dataset_files = [f for f in os.listdir("dataset") 
                             if not f.startswith("summary-") and os.path.isfile(os.path.join("dataset", f))]
            dataset_inputs, example_dropdown, load_example_btn, create_btn, prompt, response1, model1, response2, model2 = create_dataset_input()
            
            # Add status indicator to show when dataset is created
            dataset_status = gr.Markdown("*No dataset loaded*")
            
            # Load example dataset
            load_example_btn.click(
                fn=load_example_dataset,
                inputs=[example_dropdown],
                outputs=[prompt, response1, model1, response2, model2]  # Update all field values
            )

            # Save dataset to state and update status
            def create_dataset(p, r1, m1, r2, m2):
                if not p or not r1 or not r2:
                    return {}, "❌ **Error:** Please fill in at least the prompt and both responses"
                
                dataset = {
                    "entries": [
                        {"prompt": p, "response": r1, "model": m1 or "Model 1"},
                        {"prompt": p, "response": r2, "model": m2 or "Model 2"}
                    ]
                }
                return dataset, "✅ **Dataset created successfully!** You can now go to the Analysis tab"
                
            create_btn.click(
                fn=create_dataset,
                inputs=[prompt, response1, model1, response2, model2],
                outputs=[dataset_state, dataset_status]
            )
        
        # Analysis Tab
        with gr.Tab("Analysis"):
            # Use create_analysis_screen to get UI components including visualization container
            analysis_options, analysis_params, run_analysis_btn, analysis_output, ngram_n, topic_count = create_analysis_screen()
            
            # Pre-create visualization components (initially hidden)
            visualization_area_visible = gr.Checkbox(value=False, visible=False, label="Visualization Visible")
            analysis_title = gr.Markdown("## Analysis Results", visible=False)
            prompt_title = gr.Markdown(visible=False)
            models_compared = gr.Markdown(visible=False)
            
            # Container for model 1 words
            model1_title = gr.Markdown(visible=False)
            model1_words = gr.Markdown(visible=False)
            
            # Container for model 2 words
            model2_title = gr.Markdown(visible=False)
            model2_words = gr.Markdown(visible=False)
            
            # Similarity metrics
            similarity_metrics_title = gr.Markdown("### Similarity Metrics", visible=False)
            similarity_metrics = gr.Markdown(visible=False)
            
            # Status or error message area
            status_message_visible = gr.Checkbox(value=False, visible=False, label="Status Message Visible")
            status_message = gr.Markdown(visible=False)
            
            # Define a helper function to extract parameter values and run the analysis
            def run_analysis(dataset, selected_analysis, ngram_n, topic_count, user_analysis_log, *args):
                """
                Run the analysis with the selected parameters
                
                Args:
                    dataset (dict): The dataset state
                    selected_analysis (str): The selected analysis type
                    ngram_n (str or int): N value for n-gram analysis
                    topic_count (str or int): Number of topics for topic modeling
                    user_analysis_log (dict): Log of user analysis results
                    *args: Additional arguments that might be passed by Gradio
                    
                Returns:
                    tuple: Analysis results and UI component updates
                """
                try:
                    if not dataset or "entries" not in dataset or not dataset["entries"]:
                        return (
                            {},  # analysis_results_state
                            user_analysis_log,  # user_analysis_log (unchanged)
                            False,  # analysis_output visibility
                            False,  # visualization_area_visible
                            gr.update(visible=False),  # analysis_title
                            gr.update(visible=False),  # prompt_title
                            gr.update(visible=False),  # models_compared
                            gr.update(visible=False),  # model1_title
                            gr.update(visible=False),  # model1_words
                            gr.update(visible=False),  # model2_title
                            gr.update(visible=False),  # model2_words
                            gr.update(visible=False),  # similarity_metrics_title
                            gr.update(visible=False),  # similarity_metrics
                            True,  # status_message_visible
                            gr.update(visible=True, value="**Error:** No dataset loaded. Please create or load a dataset first.")  # status_message
                        )
                    
                    parameters = {
                        "bow_top": 25,  # Default fixed value for Bag of Words
                        "ngram_n": ngram_n,
                        "ngram_top": 10,  # Default fixed value for N-gram analysis
                        "topic_count": topic_count,
                        "bias_methods": ["partisan"]  # Default to partisan leaning only
                    }
                    print(f"Running analysis with selected type: {selected_analysis}")
                    print("Parameters:", parameters)
                    
                    # Process the analysis request - passing selected_analysis as a string
                    analysis_results, _ = process_analysis_request(dataset, selected_analysis, parameters)
                    
                    # If there's an error or no results
                    if not analysis_results or "analyses" not in analysis_results or not analysis_results["analyses"]:
                        return (
                            analysis_results,
                            user_analysis_log,  # user_analysis_log (unchanged)
                            False,
                            False,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            True,
                            gr.update(visible=True, value="**No results found.** Try a different analysis option.")
                        )
                    
                    # Extract information to display in components
                    prompt = list(analysis_results["analyses"].keys())[0]
                    analyses = analysis_results["analyses"][prompt]
                    
                    # Initialize visualization components visibilities and contents
                    visualization_area_visible = False
                    prompt_title_visible = False
                    prompt_title_value = ""
                    models_compared_visible = False
                    models_compared_value = ""
                    
                    model1_title_visible = False
                    model1_title_value = ""
                    model1_words_visible = False
                    model1_words_value = ""
                    
                    model2_title_visible = False
                    model2_title_value = ""
                    model2_words_visible = False
                    model2_words_value = ""
                    
                    similarity_title_visible = False
                    similarity_metrics_visible = False
                    similarity_metrics_value = ""
                    
                    # Update the user analysis log with the new results
                    updated_log = user_analysis_log.copy() if user_analysis_log else {}
                    
                    # Initialize this prompt in the log if it doesn't exist
                    if prompt not in updated_log:
                        updated_log[prompt] = {}
                    
                    # Store the analysis results in the log
                    if selected_analysis in ["Bag of Words", "N-gram Analysis", "Classifier", "Bias Detection", "Topic Modeling"]:
                        key = selected_analysis.replace(" ", "_").lower()
                        if key in analyses:
                            updated_log[prompt][selected_analysis] = {
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "result": analyses[key]
                            }
                    
                    # Check for messages from placeholder analyses
                    if "message" in analyses:
                        return (
                            analysis_results,
                            updated_log,  # Return updated log
                            False,
                            False,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            True,
                            gr.update(visible=True, value=f"**{analyses['message']}**")  # status_message
                        )
                    
                    # Process based on the selected analysis type
                    if selected_analysis == "Bag of Words" and "bag_of_words" in analyses:
                        visualization_area_visible = True
                        bow_results = analyses["bag_of_words"]
                        models = bow_results.get("models", [])
                        
                        if len(models) >= 2:
                            prompt_title_visible = True
                            prompt_title_value = f"## Analysis of Prompt: \"{prompt[:100]}...\""
                            
                            models_compared_visible = True
                            models_compared_value = f"### Comparing responses from {models[0]} and {models[1]}"
                            
                            # Extract and format information for display
                            model1_name = models[0]
                            model2_name = models[1]
                            
                            # Format important words for each model
                            important_words = bow_results.get("important_words", {})
                            
                            if model1_name in important_words:
                                model1_title_visible = True
                                model1_title_value = f"#### Top Words Used by {model1_name}"
                                
                                word_list = [f"**{item['word']}** ({item['count']})" for item in important_words[model1_name][:10]]
                                model1_words_visible = True
                                model1_words_value = ", ".join(word_list)
                            
                            if model2_name in important_words:
                                model2_title_visible = True
                                model2_title_value = f"#### Top Words Used by {model2_name}"
                                
                                word_list = [f"**{item['word']}** ({item['count']})" for item in important_words[model2_name][:10]]
                                model2_words_visible = True
                                model2_words_value = ", ".join(word_list)
                            
                            # Format similarity metrics
                            comparisons = bow_results.get("comparisons", {})
                            comparison_key = f"{model1_name} vs {model2_name}"

                            if comparison_key in comparisons:
                                metrics = comparisons[comparison_key]
                                cosine = metrics.get("cosine_similarity", 0)
                                jaccard = metrics.get("jaccard_similarity", 0)
                                semantic = metrics.get("semantic_similarity", 0)
                                common_words = metrics.get("common_word_count", 0)
                                
                                similarity_title_visible = True
                                similarity_metrics_visible = True
                                similarity_metrics_value = f"""
                                - **Cosine Similarity**: {cosine:.2f} (higher means more similar word frequency patterns)
                                - **Jaccard Similarity**: {jaccard:.2f} (higher means more word overlap)
                                - **Semantic Similarity**: {semantic:.2f} (higher means more similar meaning)
                                - **Common Words**: {common_words} words appear in both responses
                                """
                                        
                    # Check for N-gram analysis
                    elif selected_analysis == "N-gram Analysis" and "ngram_analysis" in analyses:
                        visualization_area_visible = True
                        ngram_results = analyses["ngram_analysis"]
                        models = ngram_results.get("models", [])
                        ngram_size = ngram_results.get("ngram_size", 2)
                        size_name = "Unigrams" if ngram_size == 1 else f"{ngram_size}-grams"
                        
                        if len(models) >= 2:
                            prompt_title_visible = True
                            prompt_title_value = f"## Analysis of Prompt: \"{prompt[:100]}...\""
                            
                            models_compared_visible = True
                            models_compared_value = f"### {size_name} Analysis: Comparing responses from {models[0]} and {models[1]}"
                            
                            # Extract and format information for display
                            model1_name = models[0]
                            model2_name = models[1]
                            
                            # Format important n-grams for each model
                            important_ngrams = ngram_results.get("important_ngrams", {})
                            
                            if model1_name in important_ngrams:
                                model1_title_visible = True
                                model1_title_value = f"#### Top {size_name} Used by {model1_name}"
                                
                                # Create a better formatted list of n-grams 
                                ngram_list = []
                                for item in important_ngrams[model1_name][:10]:
                                    ngram_text = item['ngram']
                                    ngram_count = item['count']
                                    ngram_list.append(f"**{ngram_text}** ({ngram_count})")
                                
                                model1_words_visible = True
                                model1_words_value = ", ".join(ngram_list)
                            
                            if model2_name in important_ngrams:
                                model2_title_visible = True
                                model2_title_value = f"#### Top {size_name} Used by {model2_name}"
                                
                                # Create a better formatted list of n-grams
                                ngram_list = []
                                for item in important_ngrams[model2_name][:10]:
                                    ngram_text = item['ngram']
                                    ngram_count = item['count']
                                    ngram_list.append(f"**{ngram_text}** ({ngram_count})")
                                
                                model2_words_visible = True
                                model2_words_value = ", ".join(ngram_list)
                            
                            # Format similarity metrics if available
                            if "comparisons" in ngram_results:
                                comparison_key = f"{model1_name} vs {model2_name}"
                                
                                if comparison_key in ngram_results["comparisons"]:
                                    metrics = ngram_results["comparisons"][comparison_key]
                                    common_count = metrics.get("common_ngram_count", 0)
                                    
                                    similarity_title_visible = True
                                    similarity_metrics_visible = True
                                    similarity_metrics_value = f"""
                                    - **Common {size_name}**: {common_count} {size_name.lower()} appear in both responses
                                    """
                            
                            # Create a new function to generate N-gram visualizations
                            def generate_ngram_visualization(important_ngrams, model1_name, model2_name):
                                plt.figure(figsize=(12, 6))
                                
                                # Process data for model 1
                                model1_data = {}
                                if model1_name in important_ngrams:
                                    for item in important_ngrams[model1_name][:10]:
                                        model1_data[item['ngram']] = item['count']
                                
                                # Process data for model 2
                                model2_data = {}
                                if model2_name in important_ngrams:
                                    for item in important_ngrams[model2_name][:10]:
                                        model2_data[item['ngram']] = item['count']
                                
                                # Plot for the first model
                                plt.subplot(1, 2, 1)
                                sorted_data1 = sorted(model1_data.items(), key=lambda x: x[1], reverse=True)[:10]
                                terms1, counts1 = zip(*sorted_data1) if sorted_data1 else ([], [])
                                
                                # Create horizontal bar chart
                                plt.barh([t[:20] + '...' if len(t) > 20 else t for t in terms1[::-1]], counts1[::-1])
                                plt.xlabel('Frequency')
                                plt.title(f'Top {size_name} Used by {model1_name}')
                                plt.tight_layout()
                                
                                # Plot for the second model
                                plt.subplot(1, 2, 2)
                                sorted_data2 = sorted(model2_data.items(), key=lambda x: x[1], reverse=True)[:10]
                                terms2, counts2 = zip(*sorted_data2) if sorted_data2 else ([], [])
                                
                                # Create horizontal bar chart
                                plt.barh([t[:20] + '...' if len(t) > 20 else t for t in terms2[::-1]], counts2[::-1])
                                plt.xlabel('Frequency')
                                plt.title(f'Top {size_name} Used by {model2_name}')
                                plt.tight_layout()
                                
                                # Save the plot to a bytes buffer
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', dpi=100)
                                buf.seek(0)
                                
                                # Convert to PIL Image
                                image = Image.open(buf)
                                return image
                            
                            # Create the visualization
                            try:
                                viz_image = generate_ngram_visualization(important_ngrams, model1_name, model2_name)
                                
                                # Convert the image to a base64 string for embedding
                                buffered = io.BytesIO()
                                viz_image.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                
                                # Append the image to the metrics_value
                                similarity_metrics_value += f"""
                                <div style="margin-top: 20px;">
                                <img src="data:image/png;base64,{img_str}" alt="N-gram visualization" style="max-width: 100%;">
                                </div>
                                """
                                similarity_metrics_visible = True
                            except Exception as viz_error:
                                print(f"Visualization error: {viz_error}")
                                # Handle the error gracefully - continue without the visualization
                    
                    # Check for Topic Modeling analysis
                    elif selected_analysis == "Topic Modeling" and "topic_modeling" in analyses:
                        visualization_area_visible = True
                        topic_results = analyses["topic_modeling"]
                        models = topic_results.get("models", [])
                        method = topic_results.get("method", "lda").upper()
                        n_topics = topic_results.get("n_topics", 3)
                        
                        if len(models) >= 2:
                            prompt_title_visible = True
                            prompt_title_value = f"## Analysis of Prompt: \"{prompt[:100]}...\""
                            
                            models_compared_visible = True
                            models_compared_value = f"### Topic Modeling Analysis ({method}, {n_topics} topics)"
                            
                            # Extract and format topic information
                            topics = topic_results.get("topics", [])
                            
                            if topics:
                                # Format topic info for display
                                topic_info = []
                                for topic in topics[:3]:  # Show first 3 topics
                                    topic_id = topic.get("id", 0)
                                    words = topic.get("words", [])[:5]  # Top 5 words per topic
                                    
                                    if words:
                                        topic_info.append(f"**Topic {topic_id+1}**: {', '.join(words)}")
                                
                                if topic_info:
                                    model1_title_visible = True
                                    model1_title_value = "#### Discovered Topics"
                                    model1_words_visible = True
                                    model1_words_value = "\n".join(topic_info)
                            
                            # Get topic distributions for models
                            model_topics = topic_results.get("model_topics", {})
                            
                            if model_topics:
                                model1_name = models[0]
                                model2_name = models[1]
                                
                                # Format topic distribution info
                                if model1_name in model_topics and model2_name in model_topics:
                                    model2_title_visible = True
                                    model2_title_value = "#### Topic Distribution"
                                    model2_words_visible = True
                                    
                                    # Simple distribution display
                                    dist1 = model_topics[model1_name]
                                    dist2 = model_topics[model2_name]
                                    
                                    model2_words_value = f"""
                                    **{model1_name}**: {', '.join([f"Topic {i+1}: {v:.2f}" for i, v in enumerate(dist1[:3])])}
                                    
                                    **{model2_name}**: {', '.join([f"Topic {i+1}: {v:.2f}" for i, v in enumerate(dist2[:3])])}
                                    """
                                        
                            # Add similarity metrics if available
                            comparisons = topic_results.get("comparisons", {})
                            if comparisons:
                                comparison_key = f"{model1_name} vs {model2_name}"
                                
                                if comparison_key in comparisons:
                                    metrics = comparisons[comparison_key]
                                    js_div = metrics.get("js_divergence", 0)
                                    
                                    similarity_title_visible = True
                                    similarity_metrics_visible = True
                                    similarity_metrics_value = f"""
                                    - **Topic Distribution Divergence**: {js_div:.4f} (lower means more similar topic distributions)
                                    """
                    
                    # Check for Classifier analysis
                    elif selected_analysis == "Classifier" and "classifier" in analyses:
                        visualization_area_visible = True
                        classifier_results = analyses["classifier"]
                        models = classifier_results.get("models", [])
                        
                        if len(models) >= 2:
                            prompt_title_visible = True
                            prompt_title_value = f"## Analysis of Prompt: \"{prompt[:100]}...\""
                            
                            models_compared_visible = True
                            models_compared_value = f"### Classifier Analysis for {models[0]} and {models[1]}"
                            
                            # Extract and format classifier information
                            model1_name = models[0]
                            model2_name = models[1]
                            
                            # Display classifications for each model
                            classifications = classifier_results.get("classifications", {})
                            
                            if classifications:
                                model1_title_visible = True
                                model1_title_value = f"#### Classification Results"
                                model1_words_visible = True
                                
                                model1_results = classifications.get(model1_name, {})
                                model2_results = classifications.get(model2_name, {})
                                
                                model1_words_value = f"""
                                **{model1_name}**:
                                - Formality: {model1_results.get('formality', 'N/A')}
                                - Sentiment: {model1_results.get('sentiment', 'N/A')}
                                - Complexity: {model1_results.get('complexity', 'N/A')}
                                
                                **{model2_name}**:
                                - Formality: {model2_results.get('formality', 'N/A')}
                                - Sentiment: {model2_results.get('sentiment', 'N/A')}
                                - Complexity: {model2_results.get('complexity', 'N/A')}
                                """
                                
                                # Show comparison
                                model2_title_visible = True
                                model2_title_value = f"#### Classification Comparison"
                                model2_words_visible = True
                                
                                differences = classifier_results.get("differences", {})
                                model2_words_value = "\n".join([
                                    f"- **{category}**: {diff}" 
                                    for category, diff in differences.items()
                                ])
                                
                                # Create visualization using matplotlib
                                
                                
                                try:
                                    # Define metrics and mappings
                                    metrics = ['Formality', 'Sentiment', 'Complexity']
                                    mapping = {
                                        'Formality': {'Informal': 1, 'Neutral': 2, 'Formal': 3},
                                        'Sentiment': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
                                        'Complexity': {'Simple': 1, 'Average': 2, 'Complex': 3}
                                    }
                                    
                                    # Get values for each model
                                    model1_vals = []
                                    model2_vals = []
                                    
                                    # Get formality value for model1
                                    formality1 = model1_results.get('formality', 'Neutral')
                                    if formality1 in mapping['Formality']:
                                        model1_vals.append(mapping['Formality'][formality1])
                                    else:
                                        model1_vals.append(2)  # Default to neutral
                                    
                                    # Get sentiment value for model1
                                    sentiment1 = model1_results.get('sentiment', 'Neutral')
                                    if sentiment1 in mapping['Sentiment']:
                                        model1_vals.append(mapping['Sentiment'][sentiment1])
                                    else:
                                        model1_vals.append(2)  # Default to neutral
                                    
                                    # Get complexity value for model1
                                    complexity1 = model1_results.get('complexity', 'Average')
                                    if complexity1 in mapping['Complexity']:
                                        model1_vals.append(mapping['Complexity'][complexity1])
                                    else:
                                        model1_vals.append(2)  # Default to average
                                    
                                    # Get formality value for model2
                                    formality2 = model2_results.get('formality', 'Neutral')
                                    if formality2 in mapping['Formality']:
                                        model2_vals.append(mapping['Formality'][formality2])
                                    else:
                                        model2_vals.append(2)  # Default to neutral
                                    
                                    # Get sentiment value for model2
                                    sentiment2 = model2_results.get('sentiment', 'Neutral')
                                    if sentiment2 in mapping['Sentiment']:
                                        model2_vals.append(mapping['Sentiment'][sentiment2])
                                    else:
                                        model2_vals.append(2)  # Default to neutral
                                    
                                    # Get complexity value for model2
                                    complexity2 = model2_results.get('complexity', 'Average')
                                    if complexity2 in mapping['Complexity']:
                                        model2_vals.append(mapping['Complexity'][complexity2])
                                    else:
                                        model2_vals.append(2)  # Default to average
                                    
                                    # Plot grouped bar chart
                                    plt.figure(figsize=(10, 6))
                                    x = range(len(metrics))
                                    width = 0.35
                                    plt.bar([p - width/2 for p in x], model1_vals, width=width, label=model1_name)
                                    plt.bar([p + width/2 for p in x], model2_vals, width=width, label=model2_name)
                                    plt.xticks(x, metrics)
                                    plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
                                    plt.ylim(0, 3.5)
                                    plt.ylabel('Level')
                                    plt.title('Comparison of Model Characteristics')
                                    plt.legend()
                                    plt.tight_layout()
                                    
                                    # Save the plot to a bytes buffer
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png', dpi=100)
                                    buf.seek(0)
                                    
                                    # Convert to PIL Image
                                    viz_image = Image.open(buf)
                                    
                                    # Convert the image to a base64 string for embedding
                                    buffered = io.BytesIO()
                                    viz_image.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode()
                                    
                                    # Append the image to the metrics_value
                                    similarity_title_visible = True
                                    similarity_metrics_visible = True
                                    similarity_metrics_value = f"""
                                    <div style="margin-top: 20px;">
                                    <img src="data:image/png;base64,{img_str}" alt="Classifier visualization" style="max-width: 100%;">
                                    </div>
                                    """
                                except Exception as viz_error:
                                    print(f"Classifier visualization error: {viz_error}")

                    # Check for Bias Detection analysis
                    elif selected_analysis == "Bias Detection" and "bias_detection" in analyses:
                        visualization_area_visible = True
                        bias_results = analyses["bias_detection"]
                        models = bias_results.get("models", [])
                        
                        if len(models) >= 2:
                            prompt_title_visible = True
                            prompt_title_value = f"## Analysis of Prompt: \"{prompt[:100]}...\""
                            
                            models_compared_visible = True
                            models_compared_value = f"### Bias Analysis: Comparing responses from {models[0]} and {models[1]}"
                            
                            # Display comparative bias results
                            model1_name = models[0]
                            model2_name = models[1]
                            
                            if "comparative" in bias_results:
                                comparative = bias_results["comparative"]
                                
                                # Format summary for display
                                model1_title_visible = True
                                model1_title_value = "#### Bias Detection Summary"
                                model1_words_visible = True
                                
                                summary_parts = []
                                
                                # Add partisan comparison (focus on partisan leaning)
                                if "partisan" in comparative:
                                    part = comparative["partisan"]
                                    is_significant = part.get("significant", False)
                                    summary_parts.append(
                                        f"**Partisan Leaning**: {model1_name} appears {part.get(model1_name, 'N/A')}, " +
                                        f"while {model2_name} appears {part.get(model2_name, 'N/A')}. " +
                                        f"({'Significant' if is_significant else 'Minor'} difference)"
                                    )
                                
                                # Add overall assessment
                                if "overall" in comparative:
                                    overall = comparative["overall"]
                                    significant = overall.get("significant_bias_difference", False)
                                    summary_parts.append(
                                        f"**Overall Assessment**: " +
                                        f"Analysis shows a {overall.get('difference', 0):.2f}/1.0 difference in bias patterns. " +
                                        f"({'Significant' if significant else 'Minor'} overall bias difference)"
                                    )
                                
                                # Combine all parts
                                model1_words_value = "\n\n".join(summary_parts)
                                
                                # Format detailed term analysis
                                if (model1_name in bias_results and "partisan" in bias_results[model1_name] and
                                    model2_name in bias_results and "partisan" in bias_results[model2_name]):
                                    
                                    model2_title_visible = True
                                    model2_title_value = "#### Partisan Term Analysis"
                                    model2_words_visible = True
                                    
                                    m1_lib = bias_results[model1_name]["partisan"].get("liberal_terms", [])
                                    m1_con = bias_results[model1_name]["partisan"].get("conservative_terms", [])
                                    m2_lib = bias_results[model2_name]["partisan"].get("liberal_terms", [])
                                    m2_con = bias_results[model2_name]["partisan"].get("conservative_terms", [])
                                    
                                    model2_words_value = f"""
                                    **{model1_name}**:
                                    - Liberal terms: {', '.join(m1_lib) if m1_lib else 'None detected'}
                                    - Conservative terms: {', '.join(m1_con) if m1_con else 'None detected'}
                                    
                                    **{model2_name}**:
                                    - Liberal terms: {', '.join(m2_lib) if m2_lib else 'None detected'}
                                    - Conservative terms: {', '.join(m2_con) if m2_con else 'None detected'}
                                    """
                    
                    # If we don't have visualization data from any analysis
                    if not visualization_area_visible:
                        return (
                            analysis_results,
                            updated_log,  # Return updated log
                            False,
                            False,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            True,  # status_message_visible
                            gr.update(visible=True, value="**No visualization data found.** Make sure to select a valid analysis option.")
                        )

                    # Return all updated component values
                    return (
                        analysis_results,  # analysis_results_state
                        updated_log,  # user_analysis_log (updated with new results)
                        False,  # analysis_output visibility
                        True,   # visualization_area_visible
                        gr.update(visible=True),  # analysis_title
                        gr.update(visible=prompt_title_visible, value=prompt_title_value),  # prompt_title
                        gr.update(visible=models_compared_visible, value=models_compared_value),  # models_compared
                        gr.update(visible=model1_title_visible, value=model1_title_value),  # model1_title
                        gr.update(visible=model1_words_visible, value=model1_words_value),  # model1_words
                        gr.update(visible=model2_title_visible, value=model2_title_value),  # model2_title
                        gr.update(visible=model2_words_visible, value=model2_words_value),  # model2_words
                        gr.update(visible=similarity_title_visible),  # similarity_metrics_title
                        gr.update(visible=similarity_metrics_visible, value=similarity_metrics_value),  # similarity_metrics
                        False,  # status_message_visible
                        gr.update(visible=False)  # status_message
                    )
                
                except Exception as e:
                    import traceback
                    error_msg = f"Error in analysis: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    
                    return (
                        {"error": error_msg},  # analysis_results_state
                        user_analysis_log,  # Return unchanged log
                        True,  # analysis_output visibility (show raw JSON for debugging)
                        False,  # visualization_area_visible
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        True,  # status_message_visible
                        gr.update(visible=True, value=f"**Error during analysis:**\n\n```\n{str(e)}\n```")  # status_message
                    )
        
        # RoBERTa Sentiment Analysis Tab
        with gr.Tab("RoBERTa Sentiment"):
            # Create the RoBERTa analysis UI components
            run_roberta_btn, roberta_output, visualization_container, roberta_status = create_roberta_screen()
            
            # Create a container for visualization results
            with gr.Column() as roberta_viz_container:
                # create placeholder components to update
                roberta_viz_title = gr.Markdown("## RoBERTa Sentiment Analysis Results", visible=False)
                roberta_viz_content = gr.HTML("", visible=False)
            
            # Function to run RoBERTa sentiment analysis
            def run_roberta_analysis(dataset, existing_log):
                try:
                    print("Starting run_roberta_analysis function")
                    if not dataset or "entries" not in dataset or not dataset["entries"]:
                        return (
                            {},  # roberta_results_state
                            existing_log,  # no change to user_analysis_log
                            gr.update(visible=True, value="**Error:** No dataset loaded. Please create or load a dataset first."),  # roberta_status
                            gr.update(visible=False),  # roberta_output
                            gr.update(visible=False),  # roberta_viz_title
                            gr.update(visible=False)   # roberta_viz_content
                        )
                    
                    print(f"Running RoBERTa sentiment analysis with sentence-level, style=")
                    
                    # Process the analysis request
                    roberta_results = process_roberta_request(dataset)

                    print(f"RoBERTa results obtained. Size: {len(str(roberta_results))} characters")
                    
                    # NEW: Update the user analysis log with RoBERTa results
                    updated_log = existing_log.copy() if existing_log else {}
                    
                    # Get the prompt text
                    prompt_text = None
                    if "analyses" in roberta_results:
                        prompt_text = list(roberta_results["analyses"].keys())[0] if roberta_results["analyses"] else None
                    
                    if prompt_text:
                        # Initialize this prompt in the log if it doesn't exist
                        if prompt_text not in updated_log:
                            updated_log[prompt_text] = {}
                        
                        # Store the RoBERTa results
                        if "analyses" in roberta_results and prompt_text in roberta_results["analyses"]:
                            if "roberta_sentiment" in roberta_results["analyses"][prompt_text]:
                                updated_log[prompt_text]["RoBERTa Sentiment"] = {
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "result": roberta_results["analyses"][prompt_text]["roberta_sentiment"]
                                }
                    
                    # Check if we have results
                    if "error" in roberta_results:
                        return (
                            roberta_results,  # Store in state anyway for debugging
                            updated_log,  # Return updated log
                            gr.update(visible=True, value=f"**Error:** {roberta_results['error']}"),  # roberta_status
                            gr.update(visible=False),  # Hide raw output
                            gr.update(visible=False),  # roberta_viz_title
                            gr.update(visible=False)   # roberta_viz_content
                        )
                    
                    print("About to process visualization components")
                    viz_components = process_and_visualize_sentiment_analysis(roberta_results)
                    print(f"Visualization components generated: {len(viz_components)}")
                    
                    print("Starting HTML conversion of visualization components")

                    # Convert the visualization components to HTML - OPTIMIZED VERSION 
                    print("Starting HTML conversion of visualization components")
                    html_content = "<div class='sentiment-visualization'>"
                    html_content += "<h3>Sentiment Analysis Results</h3>"
                    
                    if "analyses" in roberta_results:
                        for prompt, analyses in roberta_results["analyses"].items():
                            if "roberta_sentiment" in analyses:
                                sentiment_result = analyses["roberta_sentiment"]
                                models = sentiment_result.get("models", [])
                                
                                if len(models) >= 2:
                                    # Add overall comparison
                                    if "comparison" in sentiment_result:
                                        comparison = sentiment_result["comparison"]
                                        html_content += f"<div class='comparison-section'>"
                                        html_content += f"<p><strong>{comparison.get('difference_direction', 'Models have different sentiment patterns')}</strong></p>"
                                        html_content += f"</div>"
                                    
                                    # Add individual model results
                                    sentiment_analysis = sentiment_result.get("sentiment_analysis", {})
                                    for model in models:
                                        if model in sentiment_analysis:
                                            model_result = sentiment_analysis[model]
                                            score = model_result.get("sentiment_score", 0)
                                            label = model_result.get("label", "neutral")
                                            
                                            html_content += f"<div class='model-result'>"
                                            html_content += f"<h4>{model}</h4>"
                                            html_content += f"<p>Sentiment: <strong>{label}</strong> (Score: {score:.2f})</p>"
                                            html_content += f"</div>"

                    html_content += "</div>"
                    print("HTML conversion completed")
                    
                    # Return updated values
                    return (
                        roberta_results,  # roberta_results_state
                        updated_log,  # Return updated log
                        gr.update(visible=False),  # roberta_status (hide status message)
                        gr.update(visible=False),  # roberta_output (hide raw output)
                        gr.update(visible=True),   # roberta_viz_title (show title)
                        gr.update(visible=True, value=html_content)  # roberta_viz_content (show content)
                    )
                            
                except Exception as e:
                    import traceback
                    error_msg = f"Error in RoBERTa analysis: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    
                    return (
                        {"error": error_msg},  # roberta_results_state
                        existing_log,  # Return unchanged log
                        gr.update(visible=True, value=f"**Error during RoBERTa analysis:**\n\n```\n{str(e)}\n```"),  # roberta_status
                        gr.update(visible=False),  # Hide raw output
                        gr.update(visible=False),  # roberta_viz_title
                        gr.update(visible=False)   # roberta_viz_content
                    )
            
            # Connect the run button to the analysis function
            run_roberta_btn.click(
                fn=run_roberta_analysis,
                inputs=[dataset_state, user_analysis_log],
                outputs=[
                    roberta_results_state,
                    user_analysis_log,
                    roberta_status,
                    roberta_output,
                    roberta_viz_title,
                    roberta_viz_content
                ]
            )
        
        # Add a Summary tab
        with gr.Tab("Summary"):
            gr.Markdown("## Analysis Summaries")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Get summary files from dataset directory 
                    summary_files = [f for f in os.listdir("dataset") if f.startswith("summary-") and f.endswith(".txt")]
                    
                    # Dropdown for selecting summary file
                    summary_dropdown = gr.Dropdown(
                        choices=["YOUR DATASET RESULTS"] + summary_files,
                        label="Select Summary",
                        info="Choose a summary to display",
                        value="YOUR DATASET RESULTS"
                    )
                    
                    load_summary_btn = gr.Button("Load Summary", variant="primary")

                    summary_assistant_prompt = gr.Textbox(
                        value="Attached are the results from various NLP based comparisons between two LLM responses on the same prompt. Give your interpretation of the results.",
                        label="Analysis Assistant Prompt",
                        lines=3,
                        interactive=True,
                    )
                
                with gr.Column(scale=3):
                    summary_content = gr.Textbox(
                        label="Summary Content",
                        lines=25,
                        max_lines=50,
                        interactive=False
                    )
                    
                    summary_status = gr.Markdown("*No summary loaded*")
            
            # Function to load summary content from file or user analysis
            def load_summary_content(file_name, user_log):
                if not file_name:
                    return "", "*No summary selected*"
                
                # Handle the special "YOUR DATASET RESULTS" option
                if file_name == "YOUR DATASET RESULTS":
                    if not user_log or not any(user_log.values()):
                        return "", "**No analysis results available.** Run some analyses in the Analysis tab first."
                    
                    # Format the user analysis log as text
                    content = "# YOUR DATASET ANALYSIS RESULTS\n\n"
                    
                    for prompt, analyses in user_log.items():
                        content += f"## Analysis of Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
                        
                        if not analyses:
                            content += "_No analyses run for this prompt._\n\n"
                            continue
                        
                        # Order the analyses in a specific sequence
                        analysis_order = ["Bag of Words", "N-gram Analysis", "Classifier", "Bias Detection", "RoBERTa Sentiment"]
                        
                        for analysis_type in analysis_order:
                            if analysis_type in analyses:
                                analysis_data = analyses[analysis_type]
                                timestamp = analysis_data.get("timestamp", "")
                                result = analysis_data.get("result", {})
                                
                                content += f"### {analysis_type} ({timestamp})\n\n"
                                
                                # Format based on analysis type
                                if analysis_type == "Bag of Words":
                                    models = result.get("models", [])
                                    if len(models) >= 2:
                                        content += f"Comparing responses from {models[0]} and {models[1]}\n\n"
                                        
                                        # Add important words for each model
                                        important_words = result.get("important_words", {})
                                        for model_name in models:
                                            if model_name in important_words:
                                                content += f"Top Words Used by {model_name}\n"
                                                word_list = [f"{item['word']} ({item['count']})" for item in important_words[model_name][:10]]
                                                content += ", ".join(word_list) + "\n\n"
                                        
                                        # Add similarity metrics
                                        comparisons = result.get("comparisons", {})
                                        comparison_key = f"{models[0]} vs {models[1]}"
                                        if comparison_key in comparisons:
                                            metrics = comparisons[comparison_key]
                                            content += "Similarity Metrics\n"
                                            content += f"Cosine Similarity: {metrics.get('cosine_similarity', 0):.2f} (higher means more similar word frequency patterns)\n"
                                            content += f"Jaccard Similarity: {metrics.get('jaccard_similarity', 0):.2f} (higher means more word overlap)\n"
                                            content += f"Semantic Similarity: {metrics.get('semantic_similarity', 0):.2f} (higher means more similar meaning)\n"
                                            content += f"Common Words: {metrics.get('common_word_count', 0)} words appear in both responses\n\n"
                                
                                elif analysis_type == "N-gram Analysis":
                                    models = result.get("models", [])
                                    ngram_size = result.get("ngram_size", 2)
                                    size_name = "Unigrams" if ngram_size == 1 else f"{ngram_size}-grams"
                                    
                                    if len(models) >= 2:
                                        content += f"{size_name} Analysis: Comparing responses from {models[0]} and {models[1]}\n\n"
                                        
                                        # Add important n-grams for each model
                                        important_ngrams = result.get("important_ngrams", {})
                                        for model_name in models:
                                            if model_name in important_ngrams:
                                                content += f"Top {size_name} Used by {model_name}\n"
                                                ngram_list = [f"{item['ngram']} ({item['count']})" for item in important_ngrams[model_name][:10]]
                                                content += ", ".join(ngram_list) + "\n\n"
                                        
                                        # Add similarity metrics
                                        if "comparisons" in result:
                                            comparison_key = f"{models[0]} vs {models[1]}"
                                            if comparison_key in result["comparisons"]:
                                                metrics = result["comparisons"][comparison_key]
                                                content += "Similarity Metrics\n"
                                                content += f"Common {size_name}: {metrics.get('common_ngram_count', 0)} {size_name.lower()} appear in both responses\n\n"
                                
                                elif analysis_type == "Classifier":
                                    models = result.get("models", [])
                                    if len(models) >= 2:
                                        content += f"Classifier Analysis for {models[0]} and {models[1]}\n\n"
                                        
                                        # Add classification results
                                        classifications = result.get("classifications", {})
                                        if classifications:
                                            content += "Classification Results\n"
                                            for model_name in models:
                                                if model_name in classifications:
                                                    model_results = classifications[model_name]
                                                    content += f"{model_name}:\n"
                                                    content += f"- Formality: {model_results.get('formality', 'N/A')}\n"
                                                    content += f"- Sentiment: {model_results.get('sentiment', 'N/A')}\n"
                                                    content += f"- Complexity: {model_results.get('complexity', 'N/A')}\n\n"
                                            
                                            # Add differences
                                            differences = result.get("differences", {})
                                            if differences:
                                                content += "Classification Comparison\n"
                                                for category, diff in differences.items():
                                                    content += f"- {category}: {diff}\n"
                                                content += "\n"
                                
                                elif analysis_type == "Bias Detection":
                                    models = result.get("models", [])
                                    if len(models) >= 2:
                                        content += f"Bias Analysis: Comparing responses from {models[0]} and {models[1]}\n\n"
                                        
                                        # Add comparative results
                                        if "comparative" in result:
                                            comparative = result["comparative"]
                                            content += "Bias Detection Summary\n"
                                            
                                            if "partisan" in comparative:
                                                part = comparative["partisan"]
                                                is_significant = part.get("significant", False)
                                                content += f"Partisan Leaning: {models[0]} appears {part.get(models[0], 'N/A')}, "
                                                content += f"while {models[1]} appears {part.get(models[1], 'N/A')}. "
                                                content += f"({'Significant' if is_significant else 'Minor'} difference)\n\n"
                                            
                                            if "overall" in comparative:
                                                overall = comparative["overall"]
                                                significant = overall.get("significant_bias_difference", False)
                                                content += f"Overall Assessment: "
                                                content += f"Analysis shows a {overall.get('difference', 0):.2f}/1.0 difference in bias patterns. "
                                                content += f"({'Significant' if significant else 'Minor'} overall bias difference)\n\n"
                                            
                                            # Add partisan terms
                                            content += "Partisan Term Analysis\n"
                                            for model_name in models:
                                                if model_name in result and "partisan" in result[model_name]:
                                                    partisan = result[model_name]["partisan"]
                                                    content += f"{model_name}:\n"
                                                    
                                                    lib_terms = partisan.get("liberal_terms", [])
                                                    con_terms = partisan.get("conservative_terms", [])
                                                    
                                                    content += f"- Liberal terms: {', '.join(lib_terms) if lib_terms else 'None detected'}\n"
                                                    content += f"- Conservative terms: {', '.join(con_terms) if con_terms else 'None detected'}\n\n"
                                
                                elif analysis_type == "RoBERTa Sentiment":
                                    models = result.get("models", [])
                                    if len(models) >= 2:
                                        content += "Sentiment Analysis Results\n"
                                        
                                        # Add comparison info
                                        if "comparison" in result:
                                            comparison = result["comparison"]
                                            if "difference_direction" in comparison:
                                                content += f"{comparison['difference_direction']}\n\n"
                                        
                                        # Add individual model results
                                        sentiment_analysis = result.get("sentiment_analysis", {})
                                        for model_name in models:
                                            if model_name in sentiment_analysis:
                                                model_result = sentiment_analysis[model_name]
                                                score = model_result.get("sentiment_score", 0)
                                                label = model_result.get("label", "neutral")
                                                
                                                content += f"{model_name}\n"
                                                content += f"Sentiment: {label} (Score: {score:.2f})\n\n"
                    
                    return content, f"**Loaded user analysis results**"
                    
                # Regular file loading for built-in summaries
                file_path = os.path.join("dataset", file_name)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        return content, f"**Loaded summary**: {file_name}"
                    except Exception as e:
                        return "", f"**Error loading summary**: {str(e)}"
                else:
                    return "", f"**File not found**: {file_path}"
                
            def update_summary_dropdown(user_log):
                """Update summary dropdown options based on user log state"""
                choices = ["YOUR DATASET RESULTS"]
                choices.extend([f for f in os.listdir("dataset") if f.startswith("summary-") and f.endswith(".txt")])
                return gr.update(choices=choices, value="YOUR DATASET RESULTS")
            
            # Connect the load button to the function
            load_summary_btn.click(
                fn=load_summary_content,
                inputs=[summary_dropdown, user_analysis_log],
                outputs=[summary_content, summary_status]
            )
            
            # Also load summary when dropdown changes
            summary_dropdown.change(
                fn=load_summary_content,
                inputs=[summary_dropdown, user_analysis_log],
                outputs=[summary_content, summary_status]
            )
                # Add a Visuals tab for plotting graphs
        with gr.Tab("Visuals"):
            gr.Markdown("## Visualization Graphs")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Dropdown for selecting visualization type
                    viz_type = gr.Dropdown(
                        choices=["N-gram Comparison", "Word Frequency", "Sentiment Analysis"],
                        label="Visualization Type",
                        info="Select the type of visualization to display",
                        value="N-gram Comparison"
                    )
                    
                    # Button to generate visualization
                    generate_viz_btn = gr.Button("Generate Visualization", variant="primary")
                    
                with gr.Column(scale=3):
                    # Image component to display the plot
                    viz_output = gr.Image(
                        label="Visualization",
                        type="pil",
                        height=500
                    )
                    
                    viz_status = gr.Markdown("*No visualization generated*")
            
            # Function to generate and display visualizations
            def generate_visualization(viz_type, dataset, analysis_results):
                try:
                    if not dataset or "entries" not in dataset or not dataset["entries"]:
                        return None, "❌ **Error:** No dataset loaded. Please create or load a dataset first."
                    
                    # Example data (fallback when no real data is available)
                    ex_data = {
                        'attorney general': 3,
                        'social justice': 3,
                        'centrist approach': 2,
                        'climate change': 2,
                        'criminal justice': 2,
                        'gun control': 2,
                        'human rights': 2,
                        'justice issues': 2,
                        'measures like': 2,
                        'middle class': 2
                    }

                    gran_data = {
                        'political views': 3,
                        'vice president': 3,
                        'criminal justice': 2,
                        'democratic party': 2,
                        'foreign policy': 2,
                        'harris advocated': 2,
                        'lgbtq rights': 2,
                        'president harris': 2,
                        'social issues': 2,
                        '2019 proposed': 1
                    }
                    
                    # Use real data if available in analysis_results
                    model1_data = {}
                    model2_data = {}
                    model1_name = "Model 1"
                    model2_name = "Model 2"
                    
                    # Extract actual model names from dataset
                    if dataset and "entries" in dataset and len(dataset["entries"]) >= 2:
                        model1_name = dataset["entries"][0].get("model", "Model 1")
                        model2_name = dataset["entries"][1].get("model", "Model 2")
                    
                    # Try to get real data from analysis_results
                    if analysis_results and "analyses" in analysis_results:
                        for prompt, analyses in analysis_results["analyses"].items():
                            if viz_type == "N-gram Comparison" and "ngram_analysis" in analyses:
                                ngram_results = analyses["ngram_analysis"]
                                important_ngrams = ngram_results.get("important_ngrams", {})
                                
                                if model1_name in important_ngrams:
                                    model1_data = {item["ngram"]: item["count"] for item in important_ngrams[model1_name]}
                                    
                                if model2_name in important_ngrams:
                                    model2_data = {item["ngram"]: item["count"] for item in important_ngrams[model2_name]}
                                    
                            elif viz_type == "Word Frequency" and "bag_of_words" in analyses:
                                bow_results = analyses["bag_of_words"]
                                important_words = bow_results.get("important_words", {})
                                
                                if model1_name in important_words:
                                    model1_data = {item["word"]: item["count"] for item in important_words[model1_name]}
                                    
                                if model2_name in important_words:
                                    model2_data = {item["word"]: item["count"] for item in important_words[model2_name]}
                    
                    # If we couldn't get real data, use example data
                    if not model1_data:
                        model1_data = ex_data
                    if not model2_data:
                        model2_data = gran_data
                    
                    # Create the visualization
                    plt.figure(figsize=(10, 6))
                    
                    if viz_type == "N-gram Comparison" or viz_type == "Word Frequency":
                        # Plot for the first model
                        plt.subplot(1, 2, 1)
                        sorted_data1 = sorted(model1_data.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10
                        terms1, counts1 = zip(*sorted_data1) if sorted_data1 else ([], [])
                        
                        # Create horizontal bar chart
                        plt.barh([t[:20] + '...' if len(t) > 20 else t for t in terms1[::-1]], counts1[::-1])
                        plt.xlabel('Frequency')
                        plt.title(f'Harris, Top {viz_type.split()[0]}s Used by {model1_name}')
                        plt.tight_layout()
                        
                        # Plot for the second model
                        plt.subplot(1, 2, 2)
                        sorted_data2 = sorted(model2_data.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10
                        terms2, counts2 = zip(*sorted_data2) if sorted_data2 else ([], [])
                        
                        # Create horizontal bar chart
                        plt.barh([t[:20] + '...' if len(t) > 20 else t for t in terms2[::-1]], counts2[::-1])
                        plt.xlabel('Frequency')
                        plt.title(f'Harris, Top {viz_type.split()[0]}s Used by {model2_name}')
                        plt.tight_layout()
                    
                    elif viz_type == "Sentiment Analysis":
                        # Generate sentiment comparison visualization
                        # This would be populated with real data when available
                        sentiment_scores = {
                            model1_name: 0.75,  # Example score
                            model2_name: 0.25   # Example score
                        }
                        
                        # Extract real sentiment scores if available
                        if "roberta_results_state" in analysis_results:
                            roberta_results = analysis_results["roberta_results_state"]
                            if "analyses" in roberta_results:
                                for prompt, analyses in roberta_results["analyses"].items():
                                    if "roberta_sentiment" in analyses:
                                        sentiment_result = analyses["roberta_sentiment"]
                                        sentiment_analysis = sentiment_result.get("sentiment_analysis", {})
                                        
                                        if model1_name in sentiment_analysis:
                                            sentiment_scores[model1_name] = sentiment_analysis[model1_name].get("sentiment_score", 0)
                                            
                                        if model2_name in sentiment_analysis:
                                            sentiment_scores[model2_name] = sentiment_analysis[model2_name].get("sentiment_score", 0)
                        
                        # Create sentiment bar chart
                        plt.bar(list(sentiment_scores.keys()), list(sentiment_scores.values()))
                        plt.ylim(-1, 1)
                        plt.ylabel('Harris Sentiment Score (-1 to 1)')
                        plt.title('Harris Sentiment Analysis Comparison')
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add a zero line
                    
                    # Save the plot to a bytes buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    
                    # Convert plot to PIL Image
                    from PIL import Image
                    image = Image.open(buf)
                    
                    return image, f"**Generated {viz_type} visualization**"
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error generating visualization: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    return None, f"**Error:** {str(e)}"
            
            # Connect the generate button to the function
            generate_viz_btn.click(
                fn=generate_visualization,
                inputs=[viz_type, dataset_state, analysis_results_state],
                outputs=[viz_output, viz_status]
            )

        # Run analysis with proper parameters
        run_analysis_btn.click(
            fn=run_analysis,
            inputs=[dataset_state, analysis_options, ngram_n, topic_count, user_analysis_log],
            outputs=[
                analysis_results_state,
                user_analysis_log,
                analysis_output,
                visualization_area_visible,
                analysis_title,
                prompt_title,
                models_compared,
                model1_title,
                model1_words,
                model2_title,
                model2_words,
                similarity_metrics_title,
                similarity_metrics,
                status_message_visible,
                status_message
            ]
        )

        '''
        app.load(
            fn=lambda log: (
                update_summary_dropdown(log),
                load_summary_content("YOUR DATASET RESULTS", log)
            ),
            inputs=[user_analysis_log],
            outputs=[summary_dropdown, summary_content, summary_status]
        )
        '''

    return app

if __name__ == "__main__":
    # Download required NLTK resources before launching the app
    download_nltk_resources()
    
    app = create_app()
    app.launch()
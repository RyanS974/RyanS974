import gradio as gr
import json

# Import analysis modules
from processors.topic_modeling import extract_topics, compare_topics
from processors.ngram_analysis import compare_ngrams
from processors.bias_detection import compare_bias
from processors.bow_analysis import compare_bow
from processors.metrics import calculate_similarity
from processors.diff_highlighter import highlight_differences

def create_analysis_screen():
    """
    Create the analysis options screen
    
    Returns:
        tuple: (analysis_options, analysis_params, run_analysis_btn, analysis_output)
    """
    with gr.Column() as analysis_screen:
        gr.Markdown("## Analysis Options")
        gr.Markdown("Select which analyses you want to run on the LLM responses.")
        
        # Analysis selection
        with gr.Group():
            analysis_options = gr.CheckboxGroup(
                choices=[
                    "Topic Modeling",
                    "N-gram Analysis",
                    "Bias Detection",
                    "Bag of Words",
                    "Similarity Metrics",
                    "Difference Highlighting"
                ],
                value=[
                    "Topic Modeling",
                    "N-gram Analysis",
                    "Bag of Words",
                    "Similarity Metrics"
                ],
                label="Select Analyses to Run"
            )
        
        # Parameters for each analysis type
        with gr.Accordion("Analysis Parameters", open=False) as analysis_params:
            # Topic modeling parameters
            with gr.Group():
                gr.Markdown("### Topic Modeling Parameters")
                topic_count = gr.Slider(minimum=2, maximum=10, value=3, step=1, 
                                       label="Number of Topics")
            
            # N-gram parameters
            with gr.Group():
                gr.Markdown("### N-gram Parameters")
                ngram_n = gr.Radio(choices=["1", "2", "3"], value="2", 
                                  label="N-gram Size")
                ngram_top = gr.Slider(minimum=5, maximum=30, value=10, step=1, 
                                     label="Top N-grams to Display")
            
            # Bias detection parameters
            with gr.Group():
                gr.Markdown("### Bias Detection Parameters")
                bias_methods = gr.CheckboxGroup(
                    choices=["Sentiment Analysis", "Partisan Leaning", "Framing Analysis"],
                    value=["Sentiment Analysis", "Partisan Leaning"],
                    label="Bias Detection Methods"
                )
            
            # Bag of Words parameters  
            with gr.Group():
                gr.Markdown("### Bag of Words Parameters")
                bow_top = gr.Slider(minimum=10, maximum=100, value=25, step=5, 
                                   label="Top Words to Compare")
                
            # Similarity metrics parameters
            with gr.Group():
                gr.Markdown("### Similarity Metrics Parameters")
                similarity_metrics = gr.CheckboxGroup(
                    choices=["Cosine Similarity", "Jaccard Similarity", "Semantic Similarity"],
                    value=["Cosine Similarity", "Semantic Similarity"],
                    label="Similarity Metrics to Calculate"
                )
        
        # Run analysis button
        run_analysis_btn = gr.Button("Run Analysis", variant="primary", size="large")
        
        # Analysis output area
        analysis_output = gr.JSON(label="Analysis Results", visible=False)
    
    return analysis_options, analysis_params, run_analysis_btn, analysis_output

def process_analysis_request(dataset, selected_analyses, parameters):
    """
    Process the analysis request and run selected analyses
    
    Args:
        dataset (dict): The dataset containing prompts and LLM responses
        selected_analyses (list): List of selected analysis types
        parameters (dict): Parameters for each analysis type
        
    Returns:
        tuple: (analysis_results, analysis_output_display)
    """
    if not dataset or "entries" not in dataset or not dataset["entries"]:
        return {}, gr.update(visible=True, value=json.dumps({"error": "No dataset provided or dataset is empty"}, indent=2))
    
    analysis_results = {"analyses": {}}
    
    # Group responses by prompt
    prompts = {}
    for entry in dataset["entries"]:
        if entry["prompt"] not in prompts:
            prompts[entry["prompt"]] = []
        prompts[entry["prompt"]].append({
            "model": entry["model"],
            "response": entry["response"]
        })
    
    # Run selected analyses for each prompt
    for prompt, responses in prompts.items():
        analysis_results["analyses"][prompt] = {}
        
        # Extract just the text responses and model names
        response_texts = [r["response"] for r in responses]
        model_names = [r["model"] for r in responses]
        
        # Run Topic Modeling
        if "Topic Modeling" in selected_analyses:
            num_topics = parameters.get("topic_count", 3)
            topic_results = compare_topics(response_texts, model_names, num_topics)
            analysis_results["analyses"][prompt]["topic_modeling"] = topic_results
        
        # Run N-gram Analysis
        if "N-gram Analysis" in selected_analyses:
            n = int(parameters.get("ngram_n", 2))
            top_n = parameters.get("ngram_top", 10)
            ngram_results = compare_ngrams(response_texts, model_names, n, top_n)
            analysis_results["analyses"][prompt]["ngram_analysis"] = ngram_results
        
        # Run Bias Detection
        if "Bias Detection" in selected_analyses:
            bias_methods = parameters.get("bias_methods", ["Sentiment Analysis", "Partisan Leaning"])
            bias_results = compare_bias(response_texts, model_names, bias_methods)
            analysis_results["analyses"][prompt]["bias_detection"] = bias_results
        
        # Run Bag of Words Analysis
        if "Bag of Words" in selected_analyses:
            top_words = parameters.get("bow_top", 25)
            bow_results = compare_bow(response_texts, model_names, top_words)
            analysis_results["analyses"][prompt]["bag_of_words"] = bow_results
        
        # Run Similarity Metrics
        if "Similarity Metrics" in selected_analyses:
            metrics = parameters.get("similarity_metrics", ["Cosine Similarity"])
            similarity_results = {}
            
            # Calculate pairwise similarities
            for i in range(len(responses)):
                for j in range(i+1, len(responses)):
                    model_pair = f"{model_names[i]} vs {model_names[j]}"
                    similarity_results[model_pair] = calculate_similarity(
                        response_texts[i], response_texts[j], metrics
                    )
            
            analysis_results["analyses"][prompt]["similarity_metrics"] = similarity_results
        
        # Run Difference Highlighting
        if "Difference Highlighting" in selected_analyses:
            diff_results = {}
            
            # Calculate pairwise differences
            for i in range(len(responses)):
                for j in range(i+1, len(responses)):
                    model_pair = f"{model_names[i]} vs {model_names[j]}"
                    diff_results[model_pair] = highlight_differences(
                        response_texts[i], response_texts[j]
                    )
            
            analysis_results["analyses"][prompt]["difference_highlighting"] = diff_results
    
    return analysis_results, gr.update(visible=True, value=json.dumps(analysis_results, indent=2))

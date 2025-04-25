import gradio as gr
import json
from visualization.bow_visualizer import process_and_visualize_analysis

# Import analysis modules
from processors.topic_modeling import compare_topics
from processors.ngram_analysis import compare_ngrams
from processors.bow_analysis import compare_bow
from processors.text_classifiers import classify_formality, classify_sentiment, classify_complexity, compare_classifications
from processors.bias_detection import compare_bias

# Add the implementation of these helper functions
def extract_important_words(text, top_n=20):
    """
    Extract the most important words from a text.
    
    Args:
        text (str): Input text
        top_n (int): Number of top words to return
        
    Returns:
        list: List of important words with their counts
    """
    # Import necessary modules
    from collections import Counter
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Make sure nltk resources are available
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    try:
        tokens = word_tokenize(text.lower())
    except:
        nltk.download('punkt')
        tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_tokens)
    
    # Get the top N words
    top_words = word_counts.most_common(top_n)
    
    # Format the result
    result = [{"word": word, "count": count} for word, count in top_words]
    
    return result

def calculate_text_similarity(text1, text2):
    """
    Calculate similarity metrics between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        dict: Similarity metrics
    """
    from processors.metrics import calculate_similarity
    
    # Calculate similarity using the metrics module
    metrics = calculate_similarity(text1, text2)
    
    # Add common word count
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    
    # Make sure nltk resources are available
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    # Simple tokenization and filtering
    words1 = set([w.lower() for w in nltk.word_tokenize(text1) 
                if w.isalpha() and w.lower() not in stop_words])
    words2 = set([w.lower() for w in nltk.word_tokenize(text2) 
                if w.isalpha() and w.lower() not in stop_words])
    
    # Calculate common words
    common_words = words1.intersection(words2)
    
    # Add to metrics
    metrics["common_word_count"] = len(common_words)
    
    return metrics

def extract_ngrams(text, n=2, top_n=10):
    """
    Extract the most common n-grams from text.
    
    Args:
        text (str): Input text
        n (int or str): Size of n-grams
        top_n (int): Number of top n-grams to return
        
    Returns:
        list: List of important n-grams with their counts
    """
    import nltk
    from nltk.util import ngrams
    from collections import Counter
    
    # Convert n to int if it's a string
    if isinstance(n, str):
        n = int(n)
    
    # Make sure nltk resources are available
    try:
        tokens = nltk.word_tokenize(text.lower())
    except:
        nltk.download('punkt')
        tokens = nltk.word_tokenize(text.lower())
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Convert n-grams to strings for easier handling
    n_gram_strings = [' '.join(gram) for gram in n_grams]
    
    # Count n-gram frequencies
    n_gram_counts = Counter(n_gram_strings)
    
    # Get the top N n-grams
    top_n_grams = n_gram_counts.most_common(top_n)
    
    # Format the result
    result = [{"ngram": ngram, "count": count} for ngram, count in top_n_grams]
    
    return result

def compare_ngrams(text1, text2, n=2):
    """
    Compare n-grams between two texts.
    
    Args:
        text1 (str or list): First text
        text2 (str or list): Second text
        n (int or str): Size of n-grams
        
    Returns:
        dict: Comparison metrics
    """
    import nltk
    from nltk.util import ngrams
    from collections import Counter
    
    # Convert n to int if it's a string
    if isinstance(n, str):
        n = int(n)
    
    # Handle list inputs by converting to strings
    if isinstance(text1, list):
        text1 = ' '.join(str(item) for item in text1)
    if isinstance(text2, list):
        text2 = ' '.join(str(item) for item in text2)
    
    # Make sure nltk resources are available
    try:
        tokens1 = nltk.word_tokenize(text1.lower())
        tokens2 = nltk.word_tokenize(text2.lower())
    except:
        nltk.download('punkt')
        tokens1 = nltk.word_tokenize(text1.lower())
        tokens2 = nltk.word_tokenize(text2.lower())
    
    # Generate n-grams
    n_grams1 = set([' '.join(gram) for gram in ngrams(tokens1, n)])
    n_grams2 = set([' '.join(gram) for gram in ngrams(tokens2, n)])
    
    # Calculate common n-grams
    common_n_grams = n_grams1.intersection(n_grams2)
    
    # Return comparison metrics
    return {
        "common_ngram_count": len(common_n_grams)
    }

def perform_topic_modeling(texts, model_names, n_topics=3):
    """
    Perform topic modeling on a list of texts.
    
    Args:
        texts (list): List of text documents
        model_names (list): Names of the models
        n_topics (int): Number of topics to extract
        
    Returns:
        dict: Topic modeling results
    """
    from processors.topic_modeling import compare_topics
    
    # Use the topic modeling processor
    result = compare_topics(texts, model_names, n_topics=n_topics)
    
    return result

def process_analysis_request(dataset, selected_analysis, parameters):
    """
    Process the analysis request based on the selected options.
    
    Args:
        dataset (dict): The input dataset
        selected_analysis (str): The selected analysis type
        parameters (dict): Additional parameters for the analysis
    
    Returns:
        tuple: A tuple containing (analysis_results, visualization_data)
    """
    if not dataset or "entries" not in dataset or not dataset["entries"]:
        return {}, None
        
    # Initialize the results structure
    results = {"analyses": {}}
    
    # Get the prompt text from the first entry
    prompt_text = dataset["entries"][0].get("prompt", "")
    if not prompt_text:
        return {"error": "No prompt found in dataset"}, None
        
    # Initialize the analysis container for this prompt
    results["analyses"][prompt_text] = {}
    
    # Get model names and responses
    model1_name = dataset["entries"][0].get("model", "Model 1")
    model2_name = dataset["entries"][1].get("model", "Model 2")
    
    model1_response = dataset["entries"][0].get("response", "")
    model2_response = dataset["entries"][1].get("response", "")
    
    # Process based on the selected analysis type
    if selected_analysis == "Bag of Words":
        # Get the top_n parameter and ensure it's an integer
        top_n = parameters.get("bow_top", 25)
        if isinstance(top_n, str):
            top_n = int(top_n)
        
        print(f"Using top_n value: {top_n}")  # Debug print
        
        # Perform Bag of Words analysis using the processor
        from processors.bow_analysis import compare_bow
        bow_results = compare_bow(
            [model1_response, model2_response],
            [model1_name, model2_name],
            top_n=top_n
        )
        results["analyses"][prompt_text]["bag_of_words"] = bow_results
        
    elif selected_analysis == "N-gram Analysis":
        # Perform N-gram analysis
        ngram_size = parameters.get("ngram_n", 2)
        if isinstance(ngram_size, str):
            ngram_size = int(ngram_size)
            
        top_n = parameters.get("ngram_top", 10)  # Using default 10
        if isinstance(top_n, str):
            top_n = int(top_n)
        
        # Use the processor from the dedicated ngram_analysis module
        from processors.ngram_analysis import compare_ngrams as ngram_processor
        ngram_results = ngram_processor(
            [model1_response, model2_response],
            [model1_name, model2_name],
            n=ngram_size,
            top_n=top_n
        )
        results["analyses"][prompt_text]["ngram_analysis"] = ngram_results
        
    elif selected_analysis == "Topic Modeling":
        # Perform topic modeling analysis
        topic_count = parameters.get("topic_count", 3)
        if isinstance(topic_count, str):
            topic_count = int(topic_count)
        
        try:
            # Import the enhanced topic modeling function
            from processors.topic_modeling import compare_topics, load_all_datasets_for_topic_modeling
            
            print("Starting topic modeling analysis...")
            
            # Get all responses from dataset directory
            all_model1_responses, all_model2_responses, dataset_model_names = load_all_datasets_for_topic_modeling()
            
            # Add current responses to the collection if they're not empty
            if model1_response.strip():
                all_model1_responses.append(model1_response)
                print(f"Added current model1 response ({len(model1_response.split())} words)")
            if model2_response.strip():
                all_model2_responses.append(model2_response)
                print(f"Added current model2 response ({len(model2_response.split())} words)")
            
            # Ensure we're using all loaded responses
            print(f"Using {len(all_model1_responses)} model1 responses and {len(all_model2_responses)} model2 responses")
            
            # If we have data, perform topic modeling with all available responses
            if all_model1_responses and all_model2_responses:
                # Calculate total word count for diagnostics
                total_words_model1 = sum(len(text.split()) for text in all_model1_responses)
                total_words_model2 = sum(len(text.split()) for text in all_model2_responses)
                print(f"Total words: Model1={total_words_model1}, Model2={total_words_model2}")
                
                topic_results = compare_topics(
                    texts_set_1=all_model1_responses, 
                    texts_set_2=all_model2_responses, 
                    n_topics=topic_count,
                    model_names=[model1_name, model2_name])  # Keep original model names for output
                
                results["analyses"][prompt_text]["topic_modeling"] = topic_results
                
                # Add helpful message about using all datasets
                results["analyses"][prompt_text]["topic_modeling"]["info"] = f"Topic modeling performed using {len(all_model1_responses)} responses from model 1 and {len(all_model2_responses)} responses from model 2 for better results."
                
                # Add corpus details to help users understand the analysis
                results["analyses"][prompt_text]["topic_modeling"]["corpus_stats"] = {
                    "model1_documents": len(all_model1_responses),
                    "model2_documents": len(all_model2_responses),
                    "model1_total_words": total_words_model1,
                    "model2_total_words": total_words_model2
                }
            else:
                # Fallback to original implementation if no data found
                print("No dataset responses loaded, falling back to current responses only")
                topic_results = compare_topics(
                    texts_set_1=[model1_response], 
                    texts_set_2=[model2_response], 
                    n_topics=topic_count,
                    model_names=[model1_name, model2_name])
                
                results["analyses"][prompt_text]["topic_modeling"] = topic_results
            
            # Add helpful message if text is very short
            if (len(model1_response.split()) < 50 or len(model2_response.split()) < 50):
                if "error" not in topic_results:
                    # Add a warning message about short text
                    results["analyses"][prompt_text]["topic_modeling"]["warning"] = "One or both texts are relatively short. Topic modeling works best with longer texts."
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Topic modeling error: {str(e)}\n{error_trace}")
            results["analyses"][prompt_text]["topic_modeling"] = {
                "models": [model1_name, model2_name],
                "error": str(e),
                "message": "Topic modeling failed. Try with longer text or different parameters."
            }
    
    elif selected_analysis == "Classifier":
        # Perform classifier analysis
        from processors.text_classifiers import classify_formality, classify_sentiment, classify_complexity, compare_classifications
        
        results["analyses"][prompt_text]["classifier"] = {
            "models": [model1_name, model2_name],
            "classifications": {
                model1_name: {
                    "formality": classify_formality(model1_response),
                    "sentiment": classify_sentiment(model1_response),
                    "complexity": classify_complexity(model1_response)
                },
                model2_name: {
                    "formality": classify_formality(model2_response),
                    "sentiment": classify_sentiment(model2_response),
                    "complexity": classify_complexity(model2_response)
                }
            },
            "differences": compare_classifications(model1_response, model2_response)
        }

    elif selected_analysis == "Bias Detection":
        try:
            # Perform bias detection analysis, always focusing on partisan leaning
            from processors.bias_detection import compare_bias
            
            bias_results = compare_bias(
                model1_response, 
                model2_response,
                model_names=[model1_name, model2_name]
            )
            
            results["analyses"][prompt_text]["bias_detection"] = bias_results
            
        except Exception as e:
            import traceback
            print(f"Bias detection error: {str(e)}\n{traceback.format_exc()}")
            results["analyses"][prompt_text]["bias_detection"] = {
                "models": [model1_name, model2_name],
                "error": str(e),
                "message": "Bias detection failed. Try with different parameters."
            }
    
    else:
        # Unknown analysis type
        results["analyses"][prompt_text]["message"] = "Please select a valid analysis type."
    
    # Return both the analysis results and a placeholder for visualization data
    return results, None


def create_analysis_screen():
    """
    Create the analysis options screen with enhanced topic modeling options
    
    Returns:
        tuple: (analysis_options, analysis_params, run_analysis_btn, analysis_output, ngram_n, topic_count)
    """
    import gradio as gr
    
    with gr.Column() as analysis_screen:
        gr.Markdown("## Analysis Options")
        gr.Markdown("Select which analysis you want to run on the LLM responses.")
        
        # Change from CheckboxGroup to Radio for analysis selection
        with gr.Group():
            analysis_options = gr.Radio(
                choices=[
                    "Bag of Words",
                    "N-gram Analysis",
                    "Bias Detection",
                    "Classifier"
                ],
                value="Bag of Words",  # Default selection
                label="Select Analysis Type"
            )
        
        # Create N-gram parameters accessible at top level
        ngram_n = gr.Radio(
            choices=["1", "2", "3"], value="2", 
            label="N-gram Size",
            visible=False
        )
        
        # Create enhanced topic modeling parameter accessible at top level
        topic_count = gr.Slider(
            minimum=2, maximum=10, value=3, step=1,
            label="Number of Topics",
            info="Choose fewer topics for shorter texts, more topics for longer texts",
            visible=False
        )
        
        # Parameters for each analysis type
        with gr.Group() as analysis_params:
            # Topic modeling parameters with enhanced options
            with gr.Group(visible=False) as topic_params:
                gr.Markdown("### Topic Modeling Parameters")
                gr.Markdown("""
                Topic modeling extracts thematic patterns from text. 
                
                For best results:
                - Use longer text samples (100+ words)
                - Adjust topic count based on text length 
                - For political content, 3-5 topics usually works well
                """)
                # We're already using topic_count defined above
            
            # N-gram parameters group (using external ngram_n)
            with gr.Group(visible=False) as ngram_params:
                gr.Markdown("### N-gram Parameters")
                # We're already using ngram_n defined above
                
            # Bias detection parameters
            with gr.Group(visible=False) as bias_params:
                gr.Markdown("### Bias Detection Parameters")
                gr.Markdown("Analysis will focus on detecting partisan leaning.")
            
            # Classifier parameters
            with gr.Group(visible=False) as classifier_params:
                gr.Markdown("### Classifier Parameters")
                gr.Markdown("Classifies responses based on formality, sentiment, and complexity")
                
            # Function to update parameter visibility based on selected analysis
            def update_params_visibility(selected):
                return {
                    topic_params: gr.update(visible=selected == "Topic Modeling"),
                    ngram_params: gr.update(visible=selected == "N-gram Analysis"),
                    bias_params: gr.update(visible=selected == "Bias Detection"),
                    classifier_params: gr.update(visible=selected == "Classifier"),
                    ngram_n: gr.update(visible=selected == "N-gram Analysis"),
                    topic_count: gr.update(visible=selected == "Topic Modeling")
                }
                
            # Set up event handler for analysis selection
            analysis_options.change(
                fn=update_params_visibility,
                inputs=[analysis_options],
                outputs=[
                    topic_params, 
                    ngram_params, 
                    bias_params, 
                    classifier_params,
                    ngram_n, 
                    topic_count
                ]
            )
        
        # Run analysis button
        run_analysis_btn = gr.Button("Run Analysis", variant="primary", size="large")
        
        # Analysis output area - hidden JSON component to store raw results
        analysis_output = gr.JSON(label="Analysis Results", visible=False)
    
    # Return the components needed by app.py
    return analysis_options, analysis_params, run_analysis_btn, analysis_output, ngram_n, topic_count
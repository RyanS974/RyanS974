"""
Enhanced topic modeling processor for comparing text responses
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from scipy.spatial import distance

def load_all_datasets_for_topic_modeling():
    """
    Load all dataset files and prepare them for topic modeling.
    Uses multiple approaches to ensure files are found.
    
    Returns:
        tuple: (all_model1_responses, all_model2_responses, all_model_names)
    """
    import os
    from pathlib import Path
    from utils.text_dataset_parser import parse_text_file
    
    all_model1_responses = []
    all_model2_responses = []
    all_model_names = set()
    
    # APPROACH 1: Try loading specific known files
    known_files = [
        "person-harris.txt",
        "person-trump.txt", 
        "topic-foreign_policy.txt",
        "topic-the_economy.txt"
    ]
    
    # Try different possible paths
    possible_paths = [
        "dataset",
        os.path.join(os.path.dirname(__file__), "..", "dataset"),
        os.path.abspath("dataset")
    ]
    
    dataset_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            dataset_dir = path
            print(f"Found dataset directory at: {path}")
            
            # Try to load each known file
            for file_name in known_files:
                file_path = os.path.join(path, file_name)
                
                if os.path.exists(file_path):
                    try:
                        print(f"Loading known dataset: {file_name}")
                        dataset = parse_text_file(file_path)
                        
                        if dataset.get("response1") and dataset.get("response2"):
                            all_model1_responses.append(dataset.get("response1"))
                            all_model2_responses.append(dataset.get("response2"))
                            
                            # Collect model names
                            if dataset.get("model1"):
                                all_model_names.add(dataset.get("model1"))
                            if dataset.get("model2"):
                                all_model_names.add(dataset.get("model2"))
                            
                            print(f"Successfully loaded {file_name}")
                    except Exception as e:
                        print(f"Error loading file {file_name}: {e}")
            
            # We've found a dataset directory, no need to check other paths
            break
    
    # Convert set to list for model names
    model_names_list = list(all_model_names)
    if len(model_names_list) < 2:
        # If we couldn't find enough model names, use defaults
        model_names_list = ["Model 1", "Model 2"]
    
    print(f"Total loaded: {len(all_model1_responses)} response1 entries and {len(all_model2_responses)} response2 entries")
    
    return all_model1_responses, all_model2_responses, model_names_list

def download_nltk_resources():
    """Download required NLTK resources if not already downloaded"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

# Ensure NLTK resources are available
download_nltk_resources()

def preprocess_text(text):
    """
    Preprocess text for topic modeling with improved tokenization and lemmatization
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits but keep spaces (fixed regex)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Reduced custom stopwords list - keep more meaningful political terms
    custom_stopwords = {'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 
                        'with', 'as', 'by', 'at', 'an', 'this', 'these', 'those'}
    
    stop_words.update(custom_stopwords)
    
    # Lemmatize tokens - CHANGED from len(token) > 3 to len(token) > 2
    # This keeps more meaningful short words like "tax", "war", "law", etc.
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def get_coherence_score(model, feature_names, doc_term_matrix):
    """
    Calculate topic coherence score (approximation of UMass coherence)
    
    Args:
        model: Topic model (LDA or NMF)
        feature_names: Feature names (words)
        doc_term_matrix: Document-term matrix
        
    Returns:
        float: Coherence score
    """
    coherence_scores = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        
        # Calculate co-occurrence for all word pairs
        word_pairs_scores = []
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                word_i = top_words[i]
                word_j = top_words[j]
                
                # Get indices of these words in feature_names
                try:
                    word_i_idx = list(feature_names).index(word_i)
                    word_j_idx = list(feature_names).index(word_j)
                
                    # Calculate co-occurrence (approximation)
                    doc_i = doc_term_matrix[:, word_i_idx].toarray().flatten()
                    doc_j = doc_term_matrix[:, word_j_idx].toarray().flatten()
                    
                    co_occur = sum(1 for x, y in zip(doc_i, doc_j) if x > 0 and y > 0)
                    word_pairs_scores.append(co_occur)
                except:
                    continue
        
        if word_pairs_scores:
            coherence_scores.append(sum(word_pairs_scores) / len(word_pairs_scores))
    
    # Average coherence across all topics
    if coherence_scores:
        return sum(coherence_scores) / len(coherence_scores)
    return 0.0

def get_top_words_per_topic(model, feature_names, n_top_words=10):
    """
    Get the top words for each topic in the model with improved word selection
    
    Args:
        model: Topic model (LDA or NMF)
        feature_names (list): Feature names (words)
        n_top_words (int): Number of top words to include per topic
        
    Returns:
        list: List of topics with their top words
    """
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx].tolist()
        
        # Normalize weights for better visualization
        total_weight = sum(top_weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in top_weights]
        else:
            normalized_weights = top_weights
        
        topic_dict = {
            "id": topic_idx,
            "words": top_words,
            "weights": normalized_weights,
            "raw_weights": top_weights
        }
        topics.append(topic_dict)
    return topics

def calculate_topic_diversity(topics):
    """
    Calculate topic diversity based on word overlap
    
    Args:
        topics (list): List of topics with their words
        
    Returns:
        float: Topic diversity score (0-1, higher is more diverse)
    """
    if not topics or len(topics) < 2:
        return 1.0  # Maximum diversity for a single topic
    
    # Calculate Jaccard distance between all topic pairs
    jaccard_distances = []
    for i in range(len(topics)):
        for j in range(i+1, len(topics)):
            words_i = set(topics[i]["words"])
            words_j = set(topics[j]["words"])
            
            # Jaccard distance = 1 - Jaccard similarity
            # Jaccard similarity = |intersection| / |union|
            intersection = len(words_i.intersection(words_j))
            union = len(words_i.union(words_j))
            
            if union > 0:
                jaccard_distance = 1 - (intersection / union)
                jaccard_distances.append(jaccard_distance)
    
    # Average Jaccard distance as diversity measure
    if jaccard_distances:
        return sum(jaccard_distances) / len(jaccard_distances)
    return 0.0

def extract_topics(texts, n_topics=3, n_top_words=10, method="lda"):
    """
    Extract topics from a list of texts with enhanced preprocessing and metrics
    
    Args:
        texts (list): List of text documents
        n_topics (int): Number of topics to extract
        n_top_words (int): Number of top words per topic
        method (str): Topic modeling method ('lda' or 'nmf')
        
    Returns:
        dict: Topic modeling results with topics and document-topic distributions
    """
    result = {
        "method": method,
        "n_topics": n_topics,
        "topics": [],
        "document_topics": []
    }
    
    # Handle empty input
    if not texts or all(not text.strip() for text in texts):
        result["error"] = "No text content to analyze"
        return result
    
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Check if we have enough content after preprocessing
    if all(not text.strip() for text in preprocessed_texts):
        result["error"] = "No meaningful content after preprocessing"
        return result
    
    # Calculate total word count (new check)
    total_words = sum(len(text.split()) for text in preprocessed_texts)
    if total_words < 50:  # Topic modeling needs sufficient text
        result["error"] = f"Not enough text content ({total_words} words) for reliable topic modeling. Topic modeling works best with longer texts."
        return result
    
    try:
        # Create document-term matrix
        if method == "nmf":
            # For NMF, use TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=1.0)
        else:
            # For LDA, use CountVectorizer
            vectorizer = CountVectorizer(max_features=1000, min_df=1, max_df=1.0)
        
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # Check if we have enough features
        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) < n_topics * 2:
            # Adjust n_topics if we don't have enough features
            original_n_topics = n_topics
            n_topics = max(2, len(feature_names) // 2)
            result["adjusted_n_topics"] = n_topics
            result["original_n_topics"] = original_n_topics
            
            # Add a warning message
            result["warning"] = f"Topic count reduced from {original_n_topics} to {n_topics} due to limited vocabulary"
        
        # Apply topic modeling
        if method == "nmf":
            # Non-negative Matrix Factorization
            model = NMF(n_components=n_topics, random_state=42, max_iter=500, 
                        alpha=0.1, l1_ratio=0.5)
        else:
            # Latent Dirichlet Allocation with better hyperparameters
            model = LatentDirichletAllocation(
                n_components=n_topics, 
                random_state=42,
                max_iter=30,
                learning_method='online',
                learning_offset=50.0,
                doc_topic_prior=0.1,
                topic_word_prior=0.01
            )
        
        topic_distribution = model.fit_transform(X)
        
        # Get top words for each topic
        result["topics"] = get_top_words_per_topic(model, feature_names, n_top_words)
        
        # Get topic distribution for each document
        for i, dist in enumerate(topic_distribution):
            # Normalize for easier comparison
            normalized_dist = dist / np.sum(dist) if np.sum(dist) > 0 else dist
            result["document_topics"].append({
                "document_id": i,
                "distribution": normalized_dist.tolist()
            })
        
        # Calculate coherence score
        result["coherence_score"] = get_coherence_score(model, feature_names, X)
        
        # Calculate topic diversity
        result["diversity_score"] = calculate_topic_diversity(result["topics"])
        
        return result
    except Exception as e:
        import traceback
        result["error"] = f"Topic modeling failed: {str(e)}"
        result["traceback"] = traceback.format_exc()
        print(f"Topic modeling error details: {traceback.format_exc()}")
        return result

def calculate_js_divergence(p, q):
    """
    Calculate Jensen-Shannon divergence between two distributions
    
    Args:
        p (list): First probability distribution
        q (list): Second probability distribution
        
    Returns:
        float: JS divergence (0-1, lower means more similar)
    """
    # Convert to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Convert to proper probability distributions
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Calculate JS divergence
    m = (p + q) / 2
    
    # Handle potential errors
    kl_pm = 0
    for pi, mi in zip(p, m):
        if pi > 0 and mi > 0:
            kl_pm += pi * np.log2(pi / mi)
    
    kl_qm = 0
    for qi, mi in zip(q, m):
        if qi > 0 and mi > 0:
            kl_qm += qi * np.log2(qi / mi)
    
    js_divergence = (kl_pm + kl_qm) / 2
    return js_divergence

def compare_topics(texts_set_1, texts_set_2, n_topics=3, n_top_words=10, method="lda", model_names=None):
    """
    Compare topics between two sets of texts with enhanced metrics
    
    Args:
        texts_set_1 (list): First list of text documents
        texts_set_2 (list): Second list of text documents
        n_topics (int): Number of topics to extract
        n_top_words (int): Number of top words per topic
        method (str): Topic modeling method ('lda' or 'nmf')
        model_names (list, optional): Names of the models being compared
        
    Returns:
        dict: Comparison results with topics from both sets and similarity metrics
    """
    # Set default model names if not provided
    if model_names is None:
        model_names = ["Model 1", "Model 2"]
    
    # Handle case where both sets are the same (e.g., comparing same document against itself)
    if texts_set_1 == texts_set_2:
        texts_set_2 = texts_set_2.copy()  # Create a copy to avoid reference issues
    
    # Combine both sets for a first check on text length
    all_texts = texts_set_1 + texts_set_2
    total_words = sum(len(text.split()) for text in all_texts)
    
    # Early length check
    if total_words < 100:  # Arbitrary threshold for very short texts
        return {
            "error": f"Combined texts are too short ({total_words} words) for reliable topic modeling. Try using texts with at least 100 words total or reduce the number of topics.",
            "method": method,
            "n_topics": n_topics,
            "models": model_names
        }
    
    # Extract topics for each set individually
    topics_set_1 = extract_topics(texts_set_1, n_topics, n_top_words, method)
    topics_set_2 = extract_topics(texts_set_2, n_topics, n_top_words, method)
    
    # Extract topics for combined set (for a common topic space)
    combined_topics = extract_topics(all_texts, n_topics, n_top_words, method)
    
    # Check for errors
    errors = []
    warnings = []
    
    if "error" in topics_set_1:
        errors.append(f"Error in {model_names[0]} analysis: {topics_set_1['error']}")
    if "warning" in topics_set_1:
        warnings.append(f"Warning in {model_names[0]} analysis: {topics_set_1['warning']}")
        
    if "error" in topics_set_2:
        errors.append(f"Error in {model_names[1]} analysis: {topics_set_2['error']}")
    if "warning" in topics_set_2:
        warnings.append(f"Warning in {model_names[1]} analysis: {topics_set_2['warning']}")
        
    if "error" in combined_topics:
        errors.append(f"Error in combined analysis: {combined_topics['error']}")
    if "warning" in combined_topics:
        warnings.append(f"Warning in combined analysis: {combined_topics['warning']}")
    
    # If we have critical errors, return early with error information
    if errors:
        return {
            "error": " | ".join(errors),
            "warnings": warnings if warnings else None,
            "method": method,
            "n_topics": n_topics,
            "models": model_names
        }
    
    # Start building the result
    result = {
        "method": method,
        "n_topics": n_topics,
        "models": model_names
    }
    
    # Add warnings if any
    if warnings:
        result["warnings"] = warnings
    
    # If n_topics was adjusted, use the adjusted value
    if "adjusted_n_topics" in topics_set_1 or "adjusted_n_topics" in topics_set_2 or "adjusted_n_topics" in combined_topics:
        result["adjusted_n_topics"] = min(
            topics_set_1.get("adjusted_n_topics", n_topics),
            topics_set_2.get("adjusted_n_topics", n_topics),
            combined_topics.get("adjusted_n_topics", n_topics)
        )
        result["original_n_topics"] = n_topics
    
    # Add topics from individual sets
    result["topics"] = combined_topics.get("topics", [])
    
    # Calculate similarity between topics if we have results from both sets
    if "topics" in topics_set_1 and "topics" in topics_set_2:
        similarity_matrix = []
        for topic1 in topics_set_1["topics"]:
            topic_similarities = []
            words1 = set(topic1["words"])
            for topic2 in topics_set_2["topics"]:
                words2 = set(topic2["words"])
                # Jaccard similarity: intersection over union
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0
                topic_similarities.append(similarity)
            similarity_matrix.append(topic_similarities)
        
        result["similarity_matrix"] = similarity_matrix
        
        # Find the best matching topic pairs
        matched_topics = []
        for i, similarities in enumerate(similarity_matrix):
            best_match_idx = np.argmax(similarities)
            matched_topics.append({
                "set1_topic_id": i,
                "set1_topic_words": topics_set_1["topics"][i]["words"],
                "set2_topic_id": best_match_idx,
                "set2_topic_words": topics_set_2["topics"][best_match_idx]["words"],
                "similarity": similarities[best_match_idx]
            })
        
        result["matched_topics"] = matched_topics
        result["average_similarity"] = np.mean([match["similarity"] for match in matched_topics])
    
    # Calculate topic distribution differences
    topic_differences = []
    if (topics_set_1.get("document_topics", []) and 
        topics_set_2.get("document_topics", [])):
        
        # Get average topic distribution for each set
        dist1 = np.mean([doc["distribution"] for doc in topics_set_1["document_topics"]], axis=0)
        dist2 = np.mean([doc["distribution"] for doc in topics_set_2["document_topics"]], axis=0)
        
        for i in range(min(len(dist1), len(dist2))):
            topic_differences.append({
                "topic_id": i,
                "model1_weight": float(dist1[i]),
                "model2_weight": float(dist2[i]),
                "difference": float(abs(dist1[i] - dist2[i]))
            })
        
        result["topic_differences"] = topic_differences
    
    # Calculate JS Divergence if we have distributions
    js_divergence = 0
    if (topics_set_1.get("document_topics", []) and 
        topics_set_2.get("document_topics", [])):
        
        # Get topic distributions
        dist1 = topics_set_1["document_topics"][0]["distribution"]
        dist2 = topics_set_2["document_topics"][0]["distribution"]
        
        # Calculate JS divergence
        js_divergence = calculate_js_divergence(dist1, dist2)
        result["js_divergence"] = js_divergence
    
    # Add model-specific topic distributions
    result["model_topics"] = {
        model_names[0]: topics_set_1.get("document_topics", [{}])[0].get("distribution", []) if topics_set_1.get("document_topics", []) else [],
        model_names[1]: topics_set_2.get("document_topics", [{}])[0].get("distribution", []) if topics_set_2.get("document_topics", []) else []
    }
    
    # Add comparison metrics
    result["comparisons"] = {
        f"{model_names[0]} vs {model_names[1]}": {
            "js_divergence": js_divergence,
            "topic_differences": topic_differences,
            "average_topic_similarity": result.get("average_similarity", 0)
        }
    }
    
    # Add coherence and diversity scores
    result["coherence_scores"] = {
        model_names[0]: topics_set_1.get("coherence_score", 0),
        model_names[1]: topics_set_2.get("coherence_score", 0),
        "combined": combined_topics.get("coherence_score", 0)
    }
    
    result["diversity_scores"] = {
        model_names[0]: topics_set_1.get("diversity_score", 0),
        model_names[1]: topics_set_2.get("diversity_score", 0),
        "combined": combined_topics.get("diversity_score", 0)
    }
    
    return result
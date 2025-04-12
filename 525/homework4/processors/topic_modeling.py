from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text for topic modeling
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back to string
    return ' '.join(tokens)

def extract_topics(texts, num_topics=3, method='lda'):
    """
    Extract main topics using topic modeling
    
    Args:
        texts (list): List of text documents
        num_topics (int): Number of topics to extract
        method (str): Method to use ('lda' or 'nmf')
        
    Returns:
        dict: Extracted topics and their keywords
    """
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Run topic modeling
    if method == 'nmf':
        # Non-negative Matrix Factorization (often works better for short texts)
        model = NMF(n_components=num_topics, random_state=42)
    else:
        # Latent Dirichlet Allocation
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    
    model.fit(tfidf_matrix)
    
    # Extract topics and keywords
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        # Get top 10 keywords for this topic
        top_keyword_indices = topic.argsort()[:-11:-1]
        top_keywords = [feature_names[i] for i in top_keyword_indices]
        
        topics[f"Topic_{topic_idx+1}"] = {
            "keywords": top_keywords,
            "weight": float(topic.sum())  # Convert to float for JSON serialization
        }
    
    # Get topic distribution for each document
    if method == 'nmf':
        doc_topic_dist = model.transform(tfidf_matrix)
    else:
        doc_topic_dist = model.transform(tfidf_matrix)
    
    # Convert to list of dictionaries for JSON serialization
    doc_topics = []
    for i, doc_dist in enumerate(doc_topic_dist):
        # Normalize to sum to 1
        doc_dist = doc_dist / doc_dist.sum() if doc_dist.sum() > 0 else doc_dist
        
        # Convert to dictionary of topic distributions
        dist = {}
        for topic_idx, weight in enumerate(doc_dist):
            dist[f"Topic_{topic_idx+1}"] = float(weight)  # Convert to float for JSON serialization
        
        doc_topics.append(dist)
    
    return {
        "topics": topics,
        "document_topics": doc_topics
    }

def compare_topics(texts, model_names, num_topics=3):
    """
    Compare topics across different model responses
    
    Args:
        texts (list): List of text responses
        model_names (list): List of model names corresponding to responses
        num_topics (int): Number of topics to extract
        
    Returns:
        dict: Comparative topic analysis
    """
    # Extract topics
    topic_results = extract_topics(texts, num_topics)
    
    # Map document topics to models
    model_topics = {}
    for i, model in enumerate(model_names):
        model_topics[model] = topic_results["document_topics"][i]
    
    # Find primary topic for each model
    model_primary_topics = {}
    for model, topics in model_topics.items():
        primary_topic = max(topics.items(), key=lambda x: x[1])
        model_primary_topics[model] = {
            "topic": primary_topic[0],
            "weight": primary_topic[1]
        }
    
    # Format for output
    result = {
        "topics": topic_results["topics"],
        "model_topics": model_topics,
        "primary_topics": model_primary_topics,
        "models": model_names
    }
    
    return result

def topic_similarity(topic1, topic2):
    """
    Calculate similarity between topics
    
    Args:
        topic1 (dict): First topic with keywords
        topic2 (dict): Second topic with keywords
        
    Returns:
        float: Similarity score
    """
    # Extract keywords
    keywords1 = set(topic1["keywords"])
    keywords2 = set(topic2["keywords"])
    
    # Calculate Jaccard similarity
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

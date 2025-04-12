import difflib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
import html

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def highlight_differences(text1, text2):
    """
    Find and highlight textual differences
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        dict: Differences analysis
    """
    # Tokenize into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    # Compare sentence by sentence
    matcher = difflib.SequenceMatcher(None, sentences1, sentences2)
    
    # Track different and similar content
    unique_to_1 = []
    unique_to_2 = []
    similar_content = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                similar_content.append(sentences1[i])
        elif tag == 'delete':
            for i in range(i1, i2):
                unique_to_1.append(sentences1[i])
        elif tag == 'insert':
            for j in range(j1, j2):
                unique_to_2.append(sentences2[j])
        elif tag == 'replace':
            for i in range(i1, i2):
                unique_to_1.append(sentences1[i])
            for j in range(j1, j2):
                unique_to_2.append(sentences2[j])
    
    # Calculate percentage of unique content
    total_sentences1 = len(sentences1)
    total_sentences2 = len(sentences2)
    
    pct_unique_1 = len(unique_to_1) / total_sentences1 if total_sentences1 > 0 else 0
    pct_unique_2 = len(unique_to_2) / total_sentences2 if total_sentences2 > 0 else 0
    
    # Calculate content overlap
    if total_sentences1 + total_sentences2 > 0:
        overlap = 2 * len(similar_content) / (total_sentences1 + total_sentences2)
    else:
        overlap = 0
    
    # Analyze word-level differences
    # First, get common sentences
    vectorizer = TfidfVectorizer(min_df=1)
    
    # Extract significant words unique to each text
    significant_words_1 = []
    significant_words_2 = []
    
    if unique_to_1 and unique_to_2:
        # Combine unique sentences for each text
        combined_1 = ' '.join(unique_to_1)
        combined_2 = ' '.join(unique_to_2)
        
        # Create TF-IDF vectors
        try:
            tfidf_matrix = vectorizer.fit_transform([combined_1, combined_2])
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract weights for each document
            weights_1 = tfidf_matrix[0].toarray()[0]
            weights_2 = tfidf_matrix[1].toarray()[0]
            
            # Get top 10 words unique to each text
            for i in range(len(feature_names)):
                if weights_1[i] > weights_2[i] * 2:  # Significantly higher in text 1
                    significant_words_1.append((feature_names[i], weights_1[i]))
                elif weights_2[i] > weights_1[i] * 2:  # Significantly higher in text 2
                    significant_words_2.append((feature_names[i], weights_2[i]))
            
            # Sort by weight and take top 10
            significant_words_1 = sorted(significant_words_1, key=lambda x: x[1], reverse=True)[:10]
            significant_words_2 = sorted(significant_words_2, key=lambda x: x[1], reverse=True)[:10]
            
            # Convert to list of words only
            significant_words_1 = [word for word, _ in significant_words_1]
            significant_words_2 = [word for word, _ in significant_words_2]
        except:
            # Fallback if TF-IDF fails
            significant_words_1 = []
            significant_words_2 = []
    
    return {
        "unique_to_first": unique_to_1,
        "unique_to_second": unique_to_2,
        "similar_content": similar_content,
        "pct_unique_first": pct_unique_1,
        "pct_unique_second": pct_unique_2,
        "content_overlap": overlap,
        "significant_words_first": significant_words_1,
        "significant_words_second": significant_words_2
    }

def extract_unique_content(texts):
    """
    Extract content unique to each text
    
    Args:
        texts (list): List of texts
        
    Returns:
        dict: Unique content for each text
    """
    n = len(texts)
    unique_content = [[] for _ in range(n)]
    
    # Compare each text with all others
    for i in range(n):
        sentences_i = sent_tokenize(texts[i])
        
        # For each sentence in this text
        for sentence in sentences_i:
            # Check if it appears in any other text
            is_unique = True
            for j in range(n):
                if i != j and sentence in texts[j]:
                    is_unique = False
                    break
            
            if is_unique:
                unique_content[i].append(sentence)
    
    return {f"text_{i+1}_unique": content for i, content in enumerate(unique_content)}

def generate_html_diff(text1, text2):
    """
    Generate HTML with highlighted differences
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        str: HTML with highlighted differences
    """
    # Split into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    # Compare sentence by sentence
    matcher = difflib.SequenceMatcher(None, sentences1, sentences2)
    
    # Create HTML with highlighted differences
    html_output = []
    
    html_output.append('<div style="display: flex; width: 100%;">')
    
    # First text column
    html_output.append('<div style="flex: 1; padding: 10px; border-right: 1px solid #ccc;">')
    html_output.append(f'<h3>Text 1</h3>')
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('delete', 'replace'):
            # Unique to text 1 - highlight in red
            for i in range(i1, i2):
                html_output.append(f'<p style="background-color: #ffdddd;">{html.escape(sentences1[i])}</p>')
        else:
            # Common or not in text 1
            for i in range(i1, i2):
                html_output.append(f'<p>{html.escape(sentences1[i])}</p>')
    
    html_output.append('</div>')
    
    # Second text column
    html_output.append('<div style="flex: 1; padding: 10px;">')
    html_output.append(f'<h3>Text 2</h3>')
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('insert', 'replace'):
            # Unique to text 2 - highlight in green
            for j in range(j1, j2):
                html_output.append(f'<p style="background-color: #ddffdd;">{html.escape(sentences2[j])}</p>')
        else:
            # Common or not in text 2
            for j in range(j1, j2):
                html_output.append(f'<p>{html.escape(sentences2[j])}</p>')
    
    html_output.append('</div>')
    html_output.append('</div>')
    
    return ''.join(html_output)

def highlight_text_differences(diff_results, model_pair=None):
    """
    Generate HTML with highlighted differences based on analysis results
    
    Args:
        diff_results (dict): Results from highlight_differences
        model_pair (str): Model pair to compare (e.g., "GPT-4 vs Claude-3")
        
    Returns:
        str: HTML with highlighted differences
    """
    if model_pair and model_pair in diff_results:
        analysis = diff_results[model_pair]
    elif model_pair:
        # If specific pair not found, return error
        return f"<p>Model pair '{model_pair}' not found in difference analysis.</p>"
    else:
        # If no pair specified, use first available
        if not diff_results:
            return "<p>No difference analysis available.</p>"
        analysis = diff_results[list(diff_results.keys())[0]]
    
    # Extract model names from pair
    if model_pair:
        model1, model2 = model_pair.split(" vs ")
    else:
        model1 = "Text 1"
        model2 = "Text 2"
    
    html_output = []
    
    # Overall statistics
    html_output.append('<div style="margin-bottom: 20px;">')
    html_output.append('<h3>Difference Analysis</h3>')
    html_output.append(f'<p><b>Content Overlap:</b> {analysis["content_overlap"]*100:.1f}%</p>')
    html_output.append(f'<p><b>Unique to {model1}:</b> {analysis["pct_unique_first"]*100:.1f}%</p>')
    html_output.append(f'<p><b>Unique to {model2}:</b> {analysis["pct_unique_second"]*100:.1f}%</p>')
    html_output.append('</div>')
    
    # Significant words
    html_output.append('<div style="display: flex; margin-bottom: 20px;">')
    
    html_output.append('<div style="flex: 1; padding: 10px;">')
    html_output.append(f'<h4>Key terms unique to {model1}:</h4>')
    if analysis["significant_words_first"]:
        html_output.append('<ul>')
        for word in analysis["significant_words_first"]:
            html_output.append(f'<li>{html.escape(word)}</li>')
        html_output.append('</ul>')
    else:
        html_output.append('<p>No significant unique terms found.</p>')
    html_output.append('</div>')
    
    html_output.append('<div style="flex: 1; padding: 10px;">')
    html_output.append(f'<h4>Key terms unique to {model2}:</h4>')
    if analysis["significant_words_second"]:
        html_output.append('<ul>')
        for word in analysis["significant_words_second"]:
            html_output.append(f'<li>{html.escape(word)}</li>')
        html_output.append('</ul>')
    else:
        html_output.append('<p>No significant unique terms found.</p>')
    html_output.append('</div>')
    
    html_output.append('</div>')
    
    # Unique content sections
    html_output.append('<div style="display: flex;">')
    
    html_output.append('<div style="flex: 1; padding: 10px; border-right: 1px solid #ccc;">')
    html_output.append(f'<h4>Content unique to {model1}:</h4>')
    if analysis["unique_to_first"]:
        for sentence in analysis["unique_to_first"]:
            html_output.append(f'<p style="background-color: #ffdddd;">{html.escape(sentence)}</p>')
    else:
        html_output.append('<p>No unique content found.</p>')
    html_output.append('</div>')
    
    html_output.append('<div style="flex: 1; padding: 10px;">')
    html_output.append(f'<h4>Content unique to {model2}:</h4>')
    if analysis["unique_to_second"]:
        for sentence in analysis["unique_to_second"]:
            html_output.append(f'<p style="background-color: #ddffdd;">{html.escape(sentence)}</p>')
    else:
        html_output.append('<p>No unique content found.</p>')
    html_output.append('</div>')
    
    html_output.append('</div>')
    
    return ''.join(html_output)

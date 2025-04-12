# Import main functions for easier access
from .topic_modeling import extract_topics, compare_topics
from .ngram_analysis import extract_ngrams, compare_ngrams, unique_ngrams
from .bias_detection import detect_sentiment, detect_partisan_lean, detect_framing_bias, compare_bias
from .bow_analysis import create_bow, compare_bow, important_words
from .metrics import calculate_similarity, calculate_diversity, calculate_complexity
from .diff_highlighter import highlight_differences, extract_unique_content

__all__ = [
    'extract_topics', 'compare_topics',
    'extract_ngrams', 'compare_ngrams', 'unique_ngrams',
    'detect_sentiment', 'detect_partisan_lean', 'detect_framing_bias', 'compare_bias',
    'create_bow', 'compare_bow', 'important_words',
    'calculate_similarity', 'calculate_diversity', 'calculate_complexity',
    'highlight_differences', 'extract_unique_content'
]

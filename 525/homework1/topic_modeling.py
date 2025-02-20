# topic_modeling.py
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import logging
from time import time

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

overview_topicmodeling = '''
Topic Modeling Analysis Overview

This phase uses Latent Dirichlet Allocation (LDA) to discover latent topics in the corpus.
The implementation:
1. Uses preprocessed tokens from the earlier stemming phase
2. Creates a gensim dictionary and corpus
3. Trains LDA models with configurable parameters
4. Evaluates topic quality
5. Provides detailed topic analysis by author
'''


def prepare_texts_for_lda(corpus):
    """
    Prepare texts for LDA by collecting stemmed tokens from all documents
    """
    print("\nPreparing texts for topic modeling...")
    documents = []
    doc_authors = []  # Keep track of which author each document belongs to

    for author in ['hgwells', 'shakespeare']:  # Process known authors
        if author in corpus:
            for doc_id, doc_data in corpus[author].items():
                if isinstance(doc_data, dict) and 'stemmed_tokens' in doc_data:
                    # Filter out very short tokens and ensure we have actual words
                    valid_tokens = [token for token in doc_data['stemmed_tokens'] if len(token) > 2]
                    if valid_tokens:
                        documents.append(valid_tokens)
                        doc_authors.append(author)

    print(f"Prepared {len(documents)} documents for topic modeling")
    return documents, doc_authors


def train_lda_model(texts, dictionary=None, corpus=None, num_topics=5, passes=15, chunksize=2000):
    """
    Train an LDA model with the given number of topics
    """
    start_time = time()

    # Create dictionary and corpus if not provided
    if dictionary is None:
        dictionary = corpora.Dictionary(texts)
        # Filter out extreme frequencies
        dictionary.filter_extremes(no_below=2, no_above=0.8)

    if corpus is None:
        corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        chunksize=chunksize,
        alpha='auto',
        per_word_topics=True
    )

    # Calculate simple topic coherence (faster than c_v)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='u_mass'  # Using u_mass instead of c_v for speed
    )
    coherence_score = coherence_model.get_coherence()

    print(f"LDA model training completed in {time() - start_time:.2f} seconds")
    print(f"Coherence score: {coherence_score:.4f}")

    return lda_model, dictionary, corpus, coherence_score


def find_optimal_topics(texts, start=3, limit=8, step=1):
    """
    Train multiple LDA models with different numbers of topics to find optimal number
    Using a smaller range and faster coherence metric
    """
    print("\nFinding optimal number of topics...")
    coherence_scores = []
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]

    for num_topics in range(start, limit, step):
        print(f"\nTraining model with {num_topics} topics...")
        model, _, _, coherence = train_lda_model(
            texts,
            dictionary=dictionary,
            corpus=corpus,
            num_topics=num_topics,
            passes=10  # Reduced passes for faster execution
        )
        coherence_scores.append(coherence)

    # Find optimal number of topics
    optimal_num_topics = start + (np.argmax(coherence_scores) * step)
    print(f"\nOptimal number of topics: {optimal_num_topics}")
    return optimal_num_topics, coherence_scores


def print_topics(lda_model, num_words=10):
    """
    Print topics with their top words and probabilities
    """
    print("\nTopic Analysis Results:")
    print("-" * 80)
    print(f"{'Topic':<10} {'Top Words (Probability)':<70}")
    print("-" * 80)

    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    for topic_id, topic_words in topics:
        # Format words and probabilities
        words_probs = [f"{word} ({prob:.3f})" for word, prob in topic_words]
        words_str = ", ".join(words_probs[:5])  # First 5 words on first line
        print(f"{topic_id:<10} {words_str}")

        # Print remaining words on subsequent lines with proper alignment
        if len(words_probs) > 5:
            remaining_words = ", ".join(words_probs[5:])
            print(f"{'':10} {remaining_words}")
        print()


def analyze_topics_by_category(lda_model, corpus, doc_authors):
    """
    Analyze average topic distribution for each author category
    """
    # Initialize dictionary to store topic distributions
    author_topic_dist = {}

    # Get topic distribution for each document
    for idx, doc_bow in enumerate(corpus):
        author = doc_authors[idx]
        topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)

        # Convert sparse topic distribution to dense array
        dense_dist = np.zeros(lda_model.num_topics)
        for topic_id, prob in topic_dist:
            dense_dist[topic_id] = prob

        # Add to author's topic distribution
        if author not in author_topic_dist:
            author_topic_dist[author] = []
        author_topic_dist[author].append(dense_dist)

    # Calculate average distribution for each author
    print("\nTop Topics by Author:")
    print("-" * 80)

    for author in author_topic_dist:
        # Calculate mean topic distribution
        mean_dist = np.mean(author_topic_dist[author], axis=0)

        # Get top topics
        top_topics = np.argsort(mean_dist)[-3:][::-1]

        print(f"\nAuthor: {author}")
        print("Top 3 topics and their probabilities:")
        for topic_id in top_topics:
            print(f"Topic {topic_id}: {mean_dist[topic_id]:.3f}")
            # Print top 5 words for this topic
            topic_words = lda_model.show_topic(topic_id, 5)
            words_str = ", ".join([f"{word} ({prob:.3f})" for word, prob in topic_words])
            print(f"Top words: {words_str}")
        print()


def run_topic_modeling(corpus, num_topics=None):
    """
    Main function to run the topic modeling pipeline
    """
    try:
        # Prepare texts
        texts, doc_authors = prepare_texts_for_lda(corpus)

        if not texts:
            print("No documents found for topic modeling")
            return

        # Find optimal number of topics if not specified
        if num_topics is None:
            num_topics, _ = find_optimal_topics(texts)

        # Train final model
        print(f"\nTraining final model with {num_topics} topics...")
        lda_model, dictionary, bow_corpus, coherence = train_lda_model(
            texts,
            num_topics=num_topics,
            passes=20  # More passes for final model
        )

        # Print topics
        print_topics(lda_model)

        # Analyze topics by category
        analyze_topics_by_category(lda_model, bow_corpus, doc_authors)

        return lda_model, dictionary, bow_corpus

    except KeyboardInterrupt:
        print("\nTopic modeling interrupted by user. Partial results may be available.")
        return None, None, None
    except Exception as e:
        print(f"\nError during topic modeling: {str(e)}")
        return None, None, None
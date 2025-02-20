# bag_of_words.py
from sklearn.feature_extraction.text import CountVectorizer

overview_bagofwords = '''
BoW Overview


'''

# bag of words, stemming (per author vectorizer)
def bag_of_words_s(corpus):
    """
    Creates bag of words representation using stemmed tokens, with a shared vocabulary
    across all documents for each author.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with bag of words under a new key 'bag_of_words'.
    """
    for author, docs in corpus.items():
        # First, collect all stemmed texts for this author
        author_texts = []
        for doc_id, doc_data in docs.items():
            if 'stemmed_tokens' in doc_data:
                author_texts.append(' '.join(doc_data['stemmed_tokens']))

        if not author_texts:
            continue

        # Create and fit vectorizer on all texts for this author
        vectorizer = CountVectorizer()
        vectorizer.fit(author_texts)

        # Store vocabulary in author's data for later reference
        docs['vocabulary'] = vectorizer.vocabulary_
        docs['feature_names'] = vectorizer.get_feature_names_out()

        # Create bag of words for each document using the shared vocabulary
        for doc_id, doc_data in docs.items():
            if 'stemmed_tokens' in doc_data:
                text = ' '.join(doc_data['stemmed_tokens'])
                bow_representation = vectorizer.transform([text])
                doc_data['bag_of_words'] = bow_representation

    print("Bag-of-Words completed.")

# bag of words based on lemmatization
def bag_of_words_l(corpus):
    """
    Creates bag of words representation using stemmed tokens, with a shared vocabulary
    across all documents for each author.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with bag of words under a new key 'bag_of_words'.
    """
    for author, docs in corpus.items():
        # First, collect all stemmed texts for this author
        author_texts = []
        for doc_id, doc_data in docs.items():
            if 'lemmatized_tokens' in doc_data:
                author_texts.append(' '.join(doc_data['lemmatized_tokens']))

        if not author_texts:
            continue

        # Create and fit vectorizer on all texts for this author
        vectorizer = CountVectorizer()
        vectorizer.fit(author_texts)

        # Store vocabulary in author's data for later reference
        docs['vocabulary'] = vectorizer.vocabulary_
        docs['feature_names'] = vectorizer.get_feature_names_out()

        # Create bag of words for each document using the shared vocabulary
        for doc_id, doc_data in docs.items():
            if 'lemmatized_tokens' in doc_data:
                text = ' '.join(doc_data['lemmatized_tokens'])
                bow_representation = vectorizer.transform([text])
                doc_data['bag_of_words'] = bow_representation

    print("Bag-of-Words completed.")

#
def diagnose_bag_of_words(corpus, doc_id=None, author_id=None):
    """
    Diagnoses and provides statistics on the bag-of-words representation in the corpus.
    Can analyze at three levels: specific document, specific author, or all authors.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :param doc_id: Optional document ID to filter results for a specific document.
    :param author_id: Optional author ID to filter results for a specific author.
    :return: None
    """

    def calculate_statistics(bow_representation, feature_names):
        """Calculate statistics from a bag of words representation"""
        bow_array = bow_representation.toarray()
        total_terms = bow_array.sum()
        vocab_size = bow_array.shape[1]  # Total possible vocabulary size

        # Count unique terms (terms that appear at least once)
        term_frequencies = bow_array.sum(axis=0)
        unique_terms = (term_frequencies > 0).sum()

        max_term_frequency = bow_array.max()
        min_term_frequency = bow_array[bow_array > 0].min() if bow_array.sum() > 0 else 0
        avg_terms_per_doc = total_terms / bow_array.shape[0]
        avg_frequency = total_terms / unique_terms if unique_terms > 0 else 0

        # Get top 10 terms
        top_indices = term_frequencies.argsort()[-10:][::-1]
        top_terms = [(feature_names[i], term_frequencies[i]) for i in top_indices]

        return {
            'total_terms': total_terms,
            'vocabulary_size': vocab_size,
            'unique_terms': unique_terms,
            'max_term_frequency': max_term_frequency,
            'min_term_frequency': min_term_frequency,
            'avg_terms_per_doc': avg_terms_per_doc,
            'avg_frequency': avg_frequency,
            'top_terms': top_terms
        }

    results = {}

    # First pass: collect and analyze bag of words data
    for author, docs in corpus.items():
        if author_id is not None and author != author_id:
            continue

        if 'feature_names' not in docs:
            print(f"Warning: No feature names found for author {author}. Skipping.")
            continue

        feature_names = docs['feature_names']
        author_docs = []

        # Collect all documents for this author
        for cur_doc_id, doc_data in docs.items():
            if isinstance(doc_data, dict):  # Skip non-document entries like 'vocabulary'
                if doc_id is None or cur_doc_id == doc_id:
                    if 'bag_of_words' in doc_data:
                        author_docs.append((cur_doc_id, doc_data['bag_of_words']))

        if not author_docs:
            continue

        # Calculate statistics for individual documents
        individual_doc_results = []
        for cur_doc_id, bow in author_docs:
            stats = calculate_statistics(bow, feature_names)
            individual_doc_results.append((cur_doc_id, stats))

        # Calculate combined statistics for author by summing bow matrices
        combined_bow = author_docs[0][1]
        for _, bow in author_docs[1:]:
            combined_bow = combined_bow + bow

        author_stats = calculate_statistics(combined_bow, feature_names)

        results[author] = {
            'individual_docs': individual_doc_results,
            'combined_stats': author_stats,
            'num_docs': len(author_docs)
        }

    # Print results based on the level of analysis requested
    if doc_id is not None:
        # Document-level analysis
        print(f"\n=== Analysis for Document ID: {doc_id} ===")
        found = False
        for author, data in results.items():
            for doc_id_, stats in data['individual_docs']:
                if doc_id_ == doc_id:
                    found = True
                    print(f"\nAuthor: {author}")
                    for key, value in stats.items():
                        if key == 'top_terms':
                            print(f"\nTop 10 most frequent terms:")
                            for term, freq in value:
                                print(f"  {term}: {freq}")
                        else:
                            print(f"{key}: {value}")
        if not found:
            print(f"No document found with ID: {doc_id}")

    elif author_id is not None:
        # Author-level analysis
        if author_id in results:
            print(f"\n=== Analysis for Author: {author_id} ===")
            data = results[author_id]
            print(f"\nNumber of documents: {data['num_docs']}")
            stats = data['combined_stats']
            for key, value in stats.items():
                if key == 'top_terms':
                    print(f"\nTop 10 most frequent terms:")
                    for term, freq in value:
                        print(f"  {term}: {freq}")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"No data found for author: {author_id}")

    else:
        # All-authors analysis
        print("\n=== Overall Corpus Analysis ===")
        print(f"Number of authors: {len(results)}")

        total_docs = sum(data['num_docs'] for data in results.values())
        print(f"Total number of documents: {total_docs}")

        for author, data in results.items():
            print(f"\n--- Author: {author} ---")
            print(f"Number of documents: {data['num_docs']}")
            stats = data['combined_stats']
            for key, value in stats.items():
                if key == 'top_terms':
                    print(f"\nTop 10 most frequent terms:")
                    for term, freq in value:
                        print(f"  {term}: {freq}")
                else:
                    print(f"{key}: {value}")

# unified authors
def bag_of_words_unified(corpus):
    """
    Creates bag of words representation using stemmed tokens, with a single unified vocabulary
    across ALL authors and documents. Used specifically for Naive Bayes analysis.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with unified bag of words under a new key 'unified_bow'.
    """
    # First, collect all stemmed texts from all authors
    all_texts = []
    for author, docs in corpus.items():
        if isinstance(docs, dict):  # Make sure we're dealing with a dictionary
            for doc_id, doc_data in docs.items():
                if isinstance(doc_data, dict) and 'stemmed_tokens' in doc_data:  # Check type and key
                    all_texts.append(' '.join(doc_data['stemmed_tokens']))

    if not all_texts:
        print("No stemmed texts found in corpus")
        return corpus

    # Create and fit vectorizer on ALL texts
    vectorizer = CountVectorizer()
    vectorizer.fit(all_texts)

    # Store global vocabulary and feature names at corpus level
    corpus['unified_vocabulary'] = vectorizer.vocabulary_
    corpus['unified_feature_names'] = vectorizer.get_feature_names_out()

    # Create bag of words for each document using the unified vocabulary
    for author, docs in corpus.items():
        if isinstance(docs, dict):  # Make sure we're dealing with a dictionary
            # Store vocabulary reference at author level
            docs['unified_vocabulary'] = corpus['unified_vocabulary']
            docs['unified_feature_names'] = corpus['unified_feature_names']

            for doc_id, doc_data in docs.items():
                if isinstance(doc_data, dict) and 'stemmed_tokens' in doc_data:
                    text = ' '.join(doc_data['stemmed_tokens'])
                    bow_representation = vectorizer.transform([text])
                    doc_data['unified_bow'] = bow_representation

    print("Unified Bag-of-Words completed.")
    print(f"Total vocabulary size: {len(corpus['unified_vocabulary'])}")
    return corpus
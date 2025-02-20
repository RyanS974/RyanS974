import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

overview_naivebayes = '''
Naive Bayes Analysis Overview

This phase uses the existing bag-of-words representation to identify characteristic words
for each author using three different approaches:

1. Count-based: Uses raw word frequencies
2. Binary: Uses word presence/absence (0/1)
3. TF-IDF: Uses term frequency-inverse document frequency weights

For each approach, we:
1. Use the existing bag-of-words representation
2. Calculate P(word|author) probabilities
3. Compute log likelihood ratios to find distinctive words
4. Rank words by how strongly they indicate each author's style

The log likelihood ratio (LLR) tells us how much more likely a word is to appear
in one author's work versus the other's.
'''


def collect_bow_statistics(corpus):
    """
    Collect and display statistics about the bag-of-words representation
    """
    print("\nBag of Words Statistics:")
    print("-" * 50)

    for author in corpus:
        if isinstance(corpus[author], dict) and 'vocabulary' in corpus[author]:
            vocab = corpus[author]['vocabulary']
            feature_names = corpus[author]['feature_names']

            print(f"\nAuthor: {author}")
            print(f"Vocabulary size: {len(vocab)}")

            # Count total words across all documents
            total_words = 0
            doc_count = 0
            for doc_id, doc_data in corpus[author].items():
                if isinstance(doc_data, dict) and 'bag_of_words' in doc_data:
                    total_words += doc_data['bag_of_words'].sum()
                    doc_count += 1

            print(f"Total documents: {doc_count}")
            print(f"Total word occurrences: {int(total_words)}")
            print(f"Average words per document: {int(total_words / doc_count)}")

            # Sample some vocabulary words
            print("\nSample vocabulary words:")
            sample_indices = np.random.choice(len(feature_names),
                                              size=min(5, len(feature_names)),
                                              replace=False)
            for idx in sample_indices:
                print(f"  {feature_names[idx]}")

def collect_word_counts(corpus):
    """
    Use unified bag-of-words counts for Naive Bayes analysis
    """
    print("\nCollecting word counts from unified bag-of-words representation...")
    probabilities = {}

    # Skip metadata keys - only process actual author data
    author_keys = ['hgwells', 'shakespeare']  # Explicitly list known authors

    for author in author_keys:
        if author in corpus and isinstance(corpus[author], dict) and 'unified_vocabulary' in corpus[author]:
            vocab = corpus[author]['unified_vocabulary']
            feature_names = corpus[author]['unified_feature_names']

            # Initialize counts array
            word_counts = np.zeros(len(feature_names))
            doc_count = 0

            # Sum up counts across all documents
            for doc_id, doc_data in corpus[author].items():
                if isinstance(doc_data, dict) and 'unified_bow' in doc_data:
                    word_counts += doc_data['unified_bow'].toarray().sum(axis=0)
                    doc_count += 1

            print(f"\nAuthor: {author}")
            print(f"Processed {doc_count} documents")
            print(f"Total word occurrences: {int(word_counts.sum())}")

            probabilities[author] = {
                'counts': word_counts,
                'raw_counts': word_counts,  # For count-based analysis, raw_counts are the same as counts
                'total_words': word_counts.sum(),
                'vocabulary': vocab,
                'feature_names': feature_names
            }

    return probabilities

def collect_word_binary(corpus):
    """
    Convert bag-of-words to binary (presence/absence) representation using unified vocabulary
    """
    print("\nConverting bag-of-words to binary representation...")
    probabilities = {}

    # Skip metadata keys - only process actual author data
    author_keys = ['hgwells', 'shakespeare']

    for author in author_keys:
        if author in corpus and isinstance(corpus[author], dict) and 'unified_vocabulary' in corpus[author]:
            vocab = corpus[author]['unified_vocabulary']
            feature_names = corpus[author]['unified_feature_names']

            # Initialize binary presence array and raw counts array
            word_presence = np.zeros(len(feature_names))
            raw_counts = np.zeros(len(feature_names))
            doc_count = 0

            # Sum up binary presence and raw counts across all documents
            for doc_id, doc_data in corpus[author].items():
                if isinstance(doc_data, dict) and 'unified_bow' in doc_data:
                    bow_array = doc_data['unified_bow'].toarray()
                    # Convert to binary (1 if present, 0 if absent)
                    binary = (bow_array > 0).astype(int)
                    word_presence += binary.sum(axis=0)
                    # Store raw counts
                    raw_counts += bow_array.sum(axis=0)
                    doc_count += 1

            print(f"\nAuthor: {author}")
            print(f"Processed {doc_count} documents")
            print(f"Total word occurrences (binary): {int(word_presence.sum())}")

            probabilities[author] = {
                'counts': word_presence,
                'total_words': doc_count,  # For binary, total is document count
                'vocabulary': vocab,
                'feature_names': feature_names,
                'raw_counts': raw_counts  # Add raw counts
            }

    return probabilities

def collect_word_tfidf(corpus):
    """
    Convert unified bag-of-words to TF-IDF representation
    """
    print("\nConverting bag-of-words to TF-IDF representation...")
    probabilities = {}
    transformer = TfidfTransformer()

    # Skip metadata keys - only process actual author data
    author_keys = ['hgwells', 'shakespeare']

    for author in author_keys:
        if author in corpus and isinstance(corpus[author], dict) and 'unified_vocabulary' in corpus[author]:
            vocab = corpus[author]['unified_vocabulary']
            feature_names = corpus[author]['unified_feature_names']

            # Collect all documents
            doc_vectors = []
            for doc_id, doc_data in corpus[author].items():
                if isinstance(doc_data, dict) and 'unified_bow' in doc_data:
                    doc_vectors.append(doc_data['unified_bow'])

            if doc_vectors:
                # Stack all document vectors
                combined_bow = doc_vectors[0]
                for vec in doc_vectors[1:]:
                    combined_bow = combined_bow + vec

                # Transform to TF-IDF
                tfidf_matrix = transformer.fit_transform(combined_bow)
                word_weights = tfidf_matrix.toarray().sum(axis=0)

                print(f"\nAuthor: {author}")
                print(f"Processed {len(doc_vectors)} documents")
                print(f"TF-IDF sum: {word_weights.sum():.2f}")

                probabilities[author] = {
                    'counts': word_weights,  # These are now TF-IDF weights
                    'total_words': word_weights.sum(),
                    'vocabulary': vocab,
                    'feature_names': feature_names,
                    'raw_counts': combined_bow.toarray().sum(axis=0)  # Keep raw counts for reference
                }

    return probabilities

def print_top_words(results):
    """
    Print the characteristic words for each author with TF-IDF scores
    """
    for category in results:
        print(f"\nMost characteristic words for: {category}")
        print(f"{'Word':<20} {'LLR':<10} {'TF-IDF':<12} {'Raw Count':<10} {'P(w|c)':<12} {'P(w|other)':<12}")
        print("-" * 80)

        for word, llr, tfidf_score, raw_count in results[category]['top_words']:
            # Get word index from feature names
            word_idx = np.where(results[category]['feature_names'] == word)[0][0]

            # Get probabilities using index
            prob_w_c = results[category]['probabilities'][word_idx]
            prob_w_other = results[category]['other_probabilities'][word_idx]

            print(
                f"{word:<20} {llr:>10.4f} {tfidf_score:>12.4f} {int(raw_count):>10d} {prob_w_c:>12.6f} {prob_w_other:>12.6f}")

        print("\nInterpretation:")
        print("- LLR: Log Likelihood Ratio based on TF-IDF weighted probabilities")
        print("- TF-IDF: Term frequency-inverse document frequency score")
        print("- Raw Count: Original number of occurrences in text")
        print("- P(w|c): Probability of word given this author (TF-IDF weighted)")
        print("- P(w|other): Probability of word given other author (TF-IDF weighted)")

def calculate_llr(probabilities, smoothing_factor=1.0):
    """
    Calculate log likelihood ratios using binary presence/absence
    """
    print("\nCalculating log likelihood ratios...")
    results = {}

    for category in probabilities:
        # Get current category data
        binary_scores = probabilities[category]['counts']
        raw_counts = probabilities[category]['raw_counts']
        feature_names = probabilities[category]['feature_names']

        # Calculate total binary occurrences
        cat_total = np.sum(binary_scores)

        # Get other category data
        other_scores = np.zeros_like(binary_scores)
        other_raw_counts = np.zeros_like(raw_counts)
        other_total = 0
        for other_cat in probabilities:
            if other_cat != category:
                other_scores += probabilities[other_cat]['counts']
                other_raw_counts += probabilities[other_cat]['raw_counts']
                other_total += np.sum(probabilities[other_cat]['counts'])

        print(f"\nCategory: {category}")
        print(f"Total binary presence sum: {cat_total:.2f}")
        print(f"Other category binary presence sum: {other_total:.2f}")

        # Calculate probabilities with smoothing using binary scores
        p_w_c = (binary_scores + smoothing_factor) / (cat_total + smoothing_factor * len(feature_names))
        p_w_co = (other_scores + smoothing_factor) / (other_total + smoothing_factor * len(feature_names))

        # Calculate log likelihood ratio
        llr = np.log(p_w_c) - np.log(p_w_co)

        # Get top 10 words by LLR
        top_indices = np.argsort(llr)[-10:][::-1]
        top_words = [(feature_names[i], llr[i], binary_scores[i], raw_counts[i])
                     for i in top_indices]

        results[category] = {
            'llr_scores': llr,
            'top_words': top_words,
            'probabilities': p_w_c,
            'other_probabilities': p_w_co,
            'feature_names': feature_names
        }

    return results

def analyze_results(results):
    """
    Provide additional analysis of the results
    """
    print("\nAnalysis of Results:")
    print("-" * 50)

    for category in results:
        print(f"\nCategory: {category}")

        # Get statistics about LLR scores
        llr_scores = results[category]['llr_scores']
        print(f"LLR Statistics:")
        print(f"  Max LLR: {llr_scores.max():.4f}")
        print(f"  Min LLR: {llr_scores.min():.4f}")
        print(f"  Mean LLR: {llr_scores.mean():.4f}")

        # Count words strongly associated with this category
        strong_indicators = np.sum(llr_scores > 1.0)
        print(f"\nStrong indicators:")
        print(f"  {strong_indicators} words have LLR > 1.0")
        print(f"  These words are at least e≈2.718 times more likely in {category}'s texts")
#!/usr/bin/env python3
# main.py, the main file of the assignment

import sys
import numpy as np
import nltk
from datashare import DataShare
import main_menu as menu
import dataset
import bag_of_words as bw
import naive_bayes as nb
import topic_modeling as tm

# main function and loop
def main():
    # data sharing module
    data_share = DataShare()
    ds = DataShare()

    # main loop
    while True:
        # print menu
        menu.print_menu()

        # get user input
        choice = input("Enter your choice: ")

        # TODO: testing
        # match statement for user input
        match choice:
            # Beginning Section
            case "1": # Assignment
                print(data_share.overview_assignment)
                print()
                input("Press enter to continue...")

            case "2": # fast pipeline (stemming)
                print("Beginning pipeline with stemming...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                dataset.stemming_corpus(ds.corpus)
                #dataset.lemmatization_corpus(ds.corpus)
                #bw.bag_of_words_s(ds.corpus)
                input("Press enter to continue...")

            case "3":  # fast pipeline (lemmatization)
                print("Beginning pipeline with lemmatization...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                #dataset.stemming_corpus(ds.corpus)
                dataset.lemmatization_corpus(ds.corpus)
                input("Press enter to continue...")

            case "4":  # Naive Bayes pipeline (stemming)
                print("Beginning Naive Bayes pipeline...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                dataset.stemming_corpus(ds.corpus)
                #bw.bag_of_words_s(ds.corpus)  # Original per-author BoW
                #bw.bag_of_words_unified(ds.corpus)  # Unified BoW for Naive Bayes
                input("Press enter to continue...")

            case "4.5":  # Bag of Words pipeline (stemming)
                print("Beginning Naive Bayes pipeline...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                dataset.stemming_corpus(ds.corpus)
                bw.bag_of_words_s(ds.corpus)  # Original per-author BoW
                bw.bag_of_words_unified(ds.corpus)  # Unified BoW for Naive Bayes
                input("Press enter to continue...")

            case "5":  # Naive Bayes pipeline (lemmatization)
                print("Beginning Naive Bayes pipeline...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                dataset.lemmatization_corpus(ds.corpus)
                bw.bag_of_words_s(ds.corpus)  # Original per-author BoW
                bw.bag_of_words_unified(ds.corpus)  # Unified BoW for Naive Bayes
                input("Press enter to continue...")
                
            case "6":  # Full Pipeline (stemming)
                print("Beginning Naive Bayes pipeline...")
                ds.corpus = dataset.load_corpus()
                dataset.remove_punctuation(ds.corpus)
                dataset.lowercase_text(ds.corpus)
                dataset.tokenize_corpus(ds.corpus)
                dataset.remove_stopwords(ds.corpus)
                dataset.stemming_corpus(ds.corpus)
                # bag of words (both per author and unified)
                bw.bag_of_words_s(ds.corpus)  # Original per-author BoW
                bw.bag_of_words_unified(ds.corpus)  # Unified BoW for Naive Bayes
                # naive bayes (count based)
                counts = nb.collect_word_counts(ds.corpus)
                results = nb.calculate_llr(counts)
                nb.print_top_words(results)
                nb.analyze_results(results)
                # topic modeling
                tm.run_topic_modeling(ds.corpus) # topic modeling
                input("Press enter to continue...")

            # Dataset Section
            case "10": # Dataset
                print(dataset.overview_dataset)
                print()
                input("Press enter to continue...")

            case "11": # load_corpus()
                print("Loading corpus...")
                ds.corpus = dataset.load_corpus()
                input("Press enter to continue...")

            case "11.1": # diagnose corpus load
                dataset.diagnose_corpus(ds.corpus)
                dataset.diagnose_corpus_strings(ds.corpus)
                dataset.print_document_snippets(ds.corpus, search_key="text")
                input("Press enter to continue...")

            case "12": # punctuation removal
                print("Removing punctuation...")
                dataset.remove_punctuation(ds.corpus)
                input("Press enter to continue...")

            case "12.1":  # diagnose punctuation removal
                dataset.print_document_snippets(ds.corpus, search_key='punctuation_removed')
                input("Press enter to continue...")

            case "13": # lowercasing
                print(" Lowercasing...")
                dataset.lowercase_text(ds.corpus)
                input("Press enter to continue...")

            case "13.1":  # diagnose lowercasing
                dataset.print_document_snippets(ds.corpus, search_key='lowercased')
                input("Press enter to continue...")

            case "14": # tokenization
                print("Tokenizing...")
                dataset.tokenize_corpus(ds.corpus)
                input("Press enter to continue...")

            case "14.1": # diagnose tokenization
                dataset.print_document_tokens(ds.corpus, search_key='tokens', doc_key=None)
                input("Press enter to continue...")

            case "15": # stopword removal
                print("Removing stopwords...")
                dataset.remove_stopwords(ds.corpus)
                input("Press enter to continue...")

            case "15.1":  # diagnose stopword removal
                dataset.print_document_tokens(ds.corpus, search_key='stopwords_removed', doc_key=None)
                input("Press enter to continue...")

            case "16": # stemming
                print("Stemming...")
                dataset.stemming_corpus(ds.corpus)
                input("Press enter to continue...")

            case "16.1":  # diagnose stemming
                dataset.print_document_tokens(ds.corpus, search_key='stemmed_tokens', doc_key=None)
                print("For HG Wells, comparing the document 001 stopwords_removed list and the stemmed_tokens list...")
                dataset.compare_lists(ds.corpus['hgwells']['001']['stopwords_removed'], ds.corpus['hgwells']['001']['stemmed_tokens'])
                #input("Press enter to continue...")

            case "17": # lemmatization
                print("Lemmatization...")
                dataset.lemmatization_corpus(ds.corpus)
                input("Press enter to continue...")

            case "17.1":  # diagnose lemmatization
                dataset.print_document_tokens(ds.corpus, search_key='lemmatized_tokens', doc_key=None)
                input("Press enter to continue...")

            case "20": # Bag of Words
                print(bw.overview_bagofwords)
                bw.bag_of_words_s(ds.corpus)
                print()
                input("Press enter to continue...")

            case "20.1": # Bag of Words
                print("Bag of words, unified and per author...")
                bw.bag_of_words_s(ds.corpus)  # Original per-author BoW
                bw.bag_of_words_unified(ds.corpus)  # Unified BoW for Naive Bayes
                input("Press enter to continue...")

            case "21":  # diagnose bag of words (document)
                #dataset.print_document_tokens(ds.corpus, search_key='bag_of_words', doc_key=None)
                doc_id2 = input("Enter document ID to process: (format 001, 002, etc...)")
                bw.diagnose_bag_of_words(ds.corpus, doc_id=doc_id2)
                input("Press enter to continue...")

            case "22":  # diagnose bag of words (author hgwells)
                # dataset.print_document_tokens(ds.corpus, search_key='bag_of_words', doc_key=None)
                bw.diagnose_bag_of_words(ds.corpus, author_id='hgwells')
                input("Press enter to continue...")

            case "23":  # diagnose bag of words (author shakespeare)
                # dataset.print_document_tokens(ds.corpus, search_key='bag_of_words', doc_key=None)
                bw.diagnose_bag_of_words(ds.corpus, author_id='shakespeare')
                input("Press enter to continue...")

            case "24":  # diagnose bag of words (both authors)
                # dataset.print_document_tokens(ds.corpus, search_key='bag_of_words', doc_key=None)
                bw.diagnose_bag_of_words(ds.corpus)
                input("Press enter to continue...")

            case "25":  # Analyze bag-of-words statistics
                print("Analyzing bag-of-words structure...")
                nb.collect_bow_statistics(ds.corpus)
                input("Press enter to continue...")

            case "26":  # Analyze unified bag-of-words structure
                print("Analyzing unified bag-of-words structure...")
                if 'unified_vocabulary' in ds.corpus:
                    print(f"\nUnified vocabulary size: {len(ds.corpus['unified_vocabulary'])}")
                    print("\nSample vocabulary words:")
                    sample_indices = np.random.choice(len(ds.corpus['unified_feature_names']),
                                                      size=min(10, len(ds.corpus['unified_feature_names'])),
                                                      replace=False)
                    for idx in sample_indices:
                        print(f"  {ds.corpus['unified_feature_names'][idx]}")
                else:
                    print("Unified bag-of-words not found. Run the Naive Bayes pipeline first.")
                input("Press enter to continue...")

            case "30": # Naive Bayes Overview
                print(nb.overview_naivebayes)
                print()
                input("Press enter to continue...")

            case "30.1": # Standard Naive Bayes
                print("This is a standard naive bayes analysis (word count divided by total words)")
                nb.standard_nb(ds.corpus)
                input("Press enter to continue...")

            case "31":  # Naive Bayes Analysis (count)
                print("Performing Naive Bayes analysis (count)...")
                if 'unified_vocabulary' not in ds.corpus:
                    print("Unified bag-of-words not found. Run the Naive Bayes pipeline first.")
                else:
                    counts = nb.collect_word_counts(ds.corpus)
                    results = nb.calculate_llr(counts)
                    nb.print_top_words(results)
                    nb.analyze_results(results)
                input("Press enter to continue...")

            case "32":  # Naive Bayes Analysis (binary)
                print("Performing Naive Bayes analysis (binary)...")
                if 'unified_vocabulary' not in ds.corpus:
                    print("Unified bag-of-words not found. Run the Naive Bayes pipeline first.")
                else:
                    counts = nb.collect_word_binary(ds.corpus)
                    results = nb.calculate_llr(counts)
                    nb.print_top_words(results)
                    nb.analyze_results(results)
                input("Press enter to continue...")

            case "33":  # Naive Bayes Analysis (tf-idf)
                print("Performing Naive Bayes analysis (tf-idf)...")
                if 'unified_vocabulary' not in ds.corpus:
                    print("Unified bag-of-words not found. Run the Naive Bayes pipeline first.")
                else:
                    counts = nb.collect_word_tfidf(ds.corpus)
                    results = nb.calculate_llr(counts)
                    nb.print_top_words(results)
                    nb.analyze_results(results)
                input("Press enter to continue...")

            case "40":  # Topic Modeling Overview
                print(tm.overview_topicmodeling)
                print()
                input("Press enter to continue...")

            case "41":  # Run Basic Topic Modeling
                print("Running basic topic modeling analysis...")
                tm.run_topic_modeling(ds.corpus)
                input("Press enter to continue...")

            case "42":  # Advanced Topic Modeling
                print("Running advanced topic modeling analysis...")

                # Prompt for number of topics (optional)
                num_topics_input = input("Enter number of topics (or press Enter to auto-detect): ").strip()
                num_topics = int(num_topics_input) if num_topics_input.isdigit() else None

                # Prompt for additional options
                min_token_length = int(input("Minimum token length (default 3): ") or 3)
                passes = int(input("Number of LDA passes (default 20): ") or 20)

                # Ask about visualization
                vis_input = input("Generate coherence visualization? (y/n, default n): ").lower().strip()
                visualization = vis_input == 'y'

                # Optional coherence metric selection
                coherence_type = input("Coherence metric (u_mass/c_v, default u_mass): ").strip() or 'u_mass'

                # Run advanced topic modeling
                results = tm.run_topic_modeling(
                    ds.corpus,
                    num_topics=num_topics,
                    min_token_length=min_token_length,
                    passes=passes,
                    coherence_type=coherence_type,
                    visualization=visualization
                )

                input("Press enter to continue...")

            case "50": # Experimentation
                print(ex.overview_experimentation)
                print()
                input("Press enter to continue...")

            case "101": # list comparison
                dataset.compare_lists(ds.corpus['hgwells']['001']['stopwords_removed'], ds.corpus['hgwells']['001']['stemmed_tokens'])

            case "quit":
                sys.exit(0)
            case _:
                print("Invalid choice. Please try again.")
                print()
                input("Press enter to continue...")

# dunder main
if __name__ == "__main__":
    main()
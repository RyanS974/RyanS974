# dataset.py
import os
import string
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from prompt_toolkit.key_binding.bindings.cpr import load_cpr_bindings

nltk.download('punkt_tab') # punkt_tab, pre-trained models that help tokenization
nltk.download('stopwords') # stopwords, list of stopwords
nltk.download('wordnet') # wordnet, lexical database that groups words into sets of synonyms to help lemmatization

overview_dataset = '''
Dataset Overview

The main steps in this section are:

- Load the dataset from files.
These are located in two categories which are of two different authors on Project Gutenberg. HG Wells and Shakespeare are the authors, with HG Wells' work being divided into chapters (each a document) from his various books, and Shakespeare's divided into scenes (each a document).  Each has 100 documents.

- Preprocess and Normalization

1. Punctuation Removal
2. Lowercasing
3. Tokenization
4. Stopword Removal
5. Stemming or Lemmatization

We will then move on to Bag of Words, Naive Bayes, Topic Modeling, and Experimentation.
'''

def diagnose_corpus(corpus):
    """
    Provides basic diagnostics for the corpus including document sizes, average size, smallest and largest size (outliers...).
    """
    print("\nCorpus Diagnostics:")
    if not corpus:
        print("The corpus is empty.")
        return

    for author, docs in corpus.items():
        if not docs:
            print(f"No documents found for author: {author}")
            continue

        doc_sizes = [doc["size"] for doc in docs.values()]  # Extract sizes from the nested dictionary
        avg_size = sum(doc_sizes) / len(doc_sizes) if doc_sizes else 0
        smallest = min(doc_sizes) if doc_sizes else 0
        largest = max(doc_sizes) if doc_sizes else 0

        print(f"\nAuthor: {author}")
        print(f"- Total Documents: {len(docs)}")
        print(f"- Average Size: {avg_size:.2f} characters")
        print(f"- Smallest Document Size: {smallest} characters")
        print(f"- Largest Document Size: {largest} characters")

        # Highlight possible outliers
        print("- Potential Outliers (very small docs):")
        outliers = [size for size in doc_sizes if size < avg_size / 10]
        for size in outliers:
            print(f"  - Document Size: {size} characters")

def diagnose_corpus_strings(corpus):
    """
    Analyzes the corpus documents to find and print the largest and smallest 10 document sizes for each author.
    """
    print("\nDetailed Document Size Analysis by Author:")

    for author, docs in corpus.items():
        if not docs:
            print(f"\nNo documents found for author: {author}")
            continue

        # Extract document sizes
        doc_sizes = [doc["size"] for doc in docs.values()]
        sorted_sizes = sorted(doc_sizes)

        # Print largest 10 sizes
        print(f"\nLargest 10 Document Sizes for Author: {author}")
        for i in range(min(10, len(sorted_sizes))):
            size = sorted_sizes[-(i + 1)]
            doc_info = next((d for d in docs.values() if d["size"] == size), None)
            if doc_info:
                print(f"  - Document: {doc_info['filename']} (Size: {size} characters)")
            else:
                print(f"  - Document Size: {size} characters (ID not found)")

        # Print smallest 10 sizes
        print(f"\nSmallest 10 Document Sizes for Author: {author}")
        for i in range(min(10, len(sorted_sizes))):
            size = sorted_sizes[i]
            doc_info = next((d for d in docs.values() if d["size"] == size), None)
            if doc_info:
                print(f"  - Document: {doc_info['filename']} (Size: {size} characters)")
            else:
                print(f"  - Document Size: {size} characters (ID not found)")

    print(corpus['hgwells']['001']['text'])

# diagnostics on the first 2 steps (and step 0 of just loaded text files)
def print_document_snippets(corpus, search_key):
    """
    Prints the first 300 characters of each document for up to three documents per author.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    """
    for author, docs in corpus.items():
        print(f"Author: {author}")
        for doc_id, doc_data in list(docs.items())[:2]:  # Limit to first two documents per author
            if search_key in doc_data:
                text_snippet = doc_data[search_key][:300]
                print(f"Document ID: {doc_id}\nText Snippet: {text_snippet}\n{'-' * 40}")
            else:
                print(f"Document ID: {doc_id}\nNo text available.\n{'-' * 40}")
        print("\n" + "=" * 80 + "\n")

# diagnostics on the last 3 steps which use lists
def print_document_tokens(corpus, search_key, doc_key=None):
    """
    Prints the first 100 tokens from a specified document for each author in the corpus.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :param doc_key: Optional key specifying which document to process from each author. If None, processes all documents.
    """
    for author, docs in corpus.items():
        print(f"\nTokens for Author: {author}\n")

        # Check if the specified doc_key exists for the author
        if doc_key is not None and doc_key not in docs:
            print(f"Document key '{doc_key}' not found for Author: {author}. Skipping this author.")
            continue

        # Process only the specified document or the first available document if doc_key is None
        for doc_id in ([doc_key] if doc_key is not None else list(docs.keys())[:1]):
            if doc_id in docs:
                doc_data = docs[doc_id]
                if f"{search_key}" in doc_data:
                    # Get the first 100 tokens or fewer if less than 100 tokens are available
                    tokens_to_print = doc_data[search_key][:100]
                    print(f"Document ID: {doc_id}")
                    print(f"{search_key}:")
                    for idx, token in enumerate(tokens_to_print, start=1):
                        print(f"{idx}. {token}")
                    print("-" * 40)
            else:
                print(f"No tokens found for Document ID: {doc_id}")

# initial step, step 0, load corpus from text files
def load_corpus():
    corpus = {}

    # Hardcoded directories for each author
    author_dirs = {
        "hgwells": [
            "hgwells/pg35/chapters", "hgwells/pg36/chapters",
            "hgwells/pg159/chapters", "hgwells/pg718/chapters",
            "hgwells/pg780/chapters", "hgwells/pg5230/chapters"
        ],
        "shakespeare": [
            "shakespeare/pg1513/scenes", "shakespeare/pg1514/scenes",
            "shakespeare/pg1515/scenes", "shakespeare/pg1519/scenes",
            "shakespeare/pg1522/scenes", "shakespeare/pg1526/scenes",
            "shakespeare/pg1533/scenes"
        ]
    }

    for author, directories in author_dirs.items():
        corpus[author] = {}  # Initialize author's dictionary
        doc_count = 0

        for dir_path in directories:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".txt"):
                        #print(f"Processing {dir_path}/{file}")
                        file_path = os.path.join(dir_path, file)
                        file_size = os.path.getsize(file_path)
                        corpus[author][f"{doc_count:03}"] = {
                            "text": open(file_path, "r", encoding="utf-8").read(),
                            "filename": file,
                            "size": file_size
                        }
                        doc_count += 1

    print("Corpus loaded.")
    return corpus

# step 1, removal of punctuation
def remove_punctuation(corpus):
    """
    Removes punctuation and smart quotes from text within a nested corpus structure.

    :param corpus: Nested dictionary representing the corpus with authors as keys,
                   containing document dictionaries as values.
    :return: Updated corpus dictionary with all punctuation and smart quotes removed.
    """
    # Define punctuation to remove, including both single and double quotes
    exclude_punctuation = set(string.punctuation) | {'\'', '"', '‘', '’', '“', '”'}

    # Create a translation table excluding the specified punctuation marks
    translation_table = str.maketrans('', '', ''.join(exclude_punctuation))

    for author, docs in corpus.items():  # Iterate through authors
        for doc_id, doc_data in docs.items():  # Iterate through documents within each author
            if 'text' in doc_data:  # Check if 'text' key exists
                # Apply the translation table to remove specified punctuation
                doc_data['punctuation_removed'] = doc_data['text'].translate(translation_table)

    print("All punctuation and smart quotes removed.")
    return corpus

# step 2, lowercasing
def lowercase_text(corpus):
    """
    Converts all text within the documents to lowercase.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with all text in lowercase.
    """
    for author, docs in corpus.items():
        for doc_id, doc_data in docs.items():
            if 'punctuation_removed' in doc_data:
                doc_data['lowercased'] = doc_data['punctuation_removed'].lower()

    print("Lowercased.")
    return corpus

# step 3, tokenization
def tokenize_corpus(corpus):
    """
    Tokenizes the text within each document of the corpus and stores tokens in a new key 'tokens' within each document's data.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with tokenized texts stored under a 'tokens' key.
    """
    for author, docs in corpus.items():
        for doc_id, doc_data in docs.items():
            if 'lowercased' in doc_data:
                # Tokenize the text
                tokens = word_tokenize(doc_data['lowercased'])
                # Store tokens in a new key 'tokens'
                doc_data['tokens'] = tokens

    print("Tokenized.")
    return corpus

# step 4, stopword removal
def remove_stopwords(corpus):
    """
    Removes stopwords from the tokens within each document of the corpus.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with stopwords removed from the 'tokens' key.
    """
    stop_words = set(stopwords.words('english'))

    for author, docs in corpus.items():
        #print(f"Processing author: {author}")
        for doc_id, doc_data in docs.items():
            if 'tokens' in doc_data:
                # Filter out stopwords
                filtered_tokens = [token for token in doc_data['tokens'] if token not in stop_words]
                # Update the 'tokens' key with the filtered list and store it under a new key 'no_stopwords'
                doc_data['stopwords_removed'] = filtered_tokens
                #print(
                    #f"Processed document ID: {doc_id}, Tokens length before removal: {len(doc_data['tokens'])}, After removal: {len(filtered_tokens)}")
            else:
                print(f"No tokens found for Document ID: {doc_id}")

    print("Stopwords removed.")
    return corpus

# step 5, stem (or step 6, not both)
def stemming_corpus(corpus):
    """
    Stems each word in the given corpus using the Porter Stemmer.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with stemmed tokens under a new key 'stemmed_tokens'.
    """
    stemmer = PorterStemmer()

    for author, docs in corpus.items():
        #print(f"Processing author: {author}")
        for doc_id, doc_data in docs.items():
            if 'stopwords_removed' in doc_data:
                # Stem the tokens
                stemmed_tokens = [stemmer.stem(word) for word in doc_data['stopwords_removed']]
                # Update the 'tokens' key with the stemmed list and store it under a new key 'stemmed_tokens'
                doc_data['stemmed_tokens'] = stemmed_tokens
                #print(
                #    f"Processed document ID: {doc_id}, Tokens length before stemming: {len(doc_data['tokens'])}, After stemming: {len(stemmed_tokens)}")
            else:
                print(f"No tokens found for Document ID: {doc_id}")

    print("Stemming completed.")
    return corpus

# step 6, lemmatization (or step 5, not both)
def lemmatization_corpus(corpus):
    """
    Lemmatizes each word in the given corpus using WordNet Lemmatizer.

    :param corpus: Nested dictionary representing the corpus with authors as keys and documents as values.
    :return: Updated corpus dictionary with lemmatized tokens under a new key 'lemmatized_tokens'.
    """
    lemmatizer = WordNetLemmatizer()

    for author, docs in corpus.items():
        # print(f"Processing author: {author}")
        for doc_id, doc_data in docs.items():
            if 'stopwords_removed' in doc_data:
                # Lemmatize the tokens
                lemmatized_tokens = [lemmatizer.lemmatize(word) for word in doc_data['stopwords_removed']]
                # Update the 'tokens' key with the lemmatized list and store it under a new key 'lemmatized_tokens'
                doc_data['lemmatized_tokens'] = lemmatized_tokens
                # print(
                #    f"Processed document ID: {doc_id}, Tokens length before lemmatization: {len(doc_data['tokens'])}, After lemmatization: {len(lemmatized_tokens)}")
            else:
                print(f"No tokens found for Document ID: {doc_id}")

    print("Lemmatization completed.")
    return corpus

def compare_lists(list1, list2):
    # Print the number of elements in each list
    print(f"List 1 has {len(list1)} elements.")
    print(f"List 2 has {len(list2)} elements.\n")

    # Check if the two lists are equal
    print("Equality check...")
    print(f"list1 == list2")
    result = list1 == list2
    print(f"The lists are {'equal' if result else 'not equal'}.")

    # Check if set(list1) is a subset of set(list2)
    print("\nSubset check...")
    print("set(list1).issubset(set(list2))")
    result = set(list1).issubset(set(list2))
    print(
        f"list1 is {'a subset' if result else 'not a subset'} of list2. ({len(list1)} elements in subset of {len(list2)} total elements from list2.)")

    # Check if set(list2) is a subset of set(list1)
    print("\nSubset check...")
    print("set(list2).issubset(set(list1))")
    result = set(list2).issubset(set(list1))
    print(
        f"list2 is {'a subset' if result else 'not a subset'} of list1. ({len(list2)} elements in subset of {len(list1)} total elements from list1.)")

    # Find the intersection of the two sets
    print("\nIntersection check...")
    print("set(list1).intersection(set(list2))")
    result = set(list1).intersection(set(list2))
    print(f"The intersection contains {len(result)} elements from both lists.")

    # Find the difference between set(list1) and set(list2)
    print("\nDifference check...")
    print("set(list1).difference(set(list2))")
    result = set(list1).difference(set(list2))
    print(f"list1 has {len(result)} elements that are not in list2.")

    # Find the difference between set(list2) and set(list1)
    print("\nDifference check...")
    print("set(list2).difference(set(list1))")
    result = set(list2).difference(set(list1))
    print(f"list2 has {len(result)} elements that are not in list1.")

    input("Press enter to continue...")

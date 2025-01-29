# file: main.py
# author: Ryan Smith, ARI 525, Homework 0.

# standard library imports
import sys
import string
from collections import Counter

# custom imports
import utility
import tokenization
import lemmatization
import stemming
import stopwords
import lowercasing

# 3rd party imports
import contractions
import matplotlib.pyplot as plt

# nltk imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# custom print functions with ANSI escape codes
def print_bold(text):
    print(f"\033[1m{text}\033[0m")

def print_red(text):
    print(f"\033[91m{text}\033[0m")

def print_green(text):
    print(f"\033[92m{text}\033[0m")

# main function
def main():
    # these are for tokenization, lemmatization / stemming, and stopwords removal
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # get command-line argument
    args = sys.argv[1:]

    debug = True # set to default True. not needed to be passed as an argument.
    input_file = None # pg1727.txt
    lowercase = False
    stem = False
    lemma = False
    stopwords_removal = False
    no_punctuation = False
    expand_contractions = False
    list_all = False

    # check for command-line arguments
    for arg in args:
        if arg.startswith("--"):
            option = arg[2:]
            if option == "lowercase":
                lowercase = True
                print(f"Lowercasing: {lowercase}")
            elif option == "stem":
                stem = True
                print(f"Stemming: {stem}")
            elif option == "lemma":
                lemma = True
                print(f"Lemmatization: {lemma}")
            elif option == "stopwords_removal":
                stopwords_removal = True
                print(f"Stopwords Removal: {stopwords_removal}")
            elif option == "no_punctuation":
                no_punctuation = True
                print(f"Removing Punctuation: {no_punctuation}")
            elif option == "expand_contractions":
                expand_contractions = True
                print(f"Expanding Contractions: {expand_contractions}")
            elif option == "debug":
                debug = True
                print(f"Debug Mode: {debug}")
            elif option == "list_all":
                list_all = True

                                                                                                                        
        else:
            input_file = arg
            print(f"Input File: {input_file}")

    # load input text file
    if not input_file:
        raise ValueError("No input file provided.")
    print_red("Input file provided.")

    # open file and read text
    with open(input_file, 'r') as f:
        text_raw = f.read().replace("\n", " ")
    print_red("File read.")
    print(f"First 500 characters of raw text:\n {text_raw[:500]}\n")

    # punctuation removal if no_punctuation is True or expand_contractions is True
    if no_punctuation or expand_contractions:
        # additional punctuation characters to remove
        additional_punctuation = '“”’‘'
        translator = str.maketrans('', '', string.punctuation + additional_punctuation)
        text_raw = text_raw.translate(translator)
        print_red("Punctuation removed.")
        print(f"First 500 characters of text after punctuation removal:\n {text_raw[:500]}\n")

    # expand contractions (from the imported 3rd party library 'contractions')
    if expand_contractions:
        text_raw = contractions.fix(text_raw)
        print_red("Contractions expanded.")
        print(f"First 500 characters of text after expanding contractions:\n {text_raw[:500]}\n")

    # lowercasing
    if lowercase:
        text_raw = text_raw.lower()
        print_red("Text lowercased.")
        print(f"First 500 characters of text after lowercasing:\n {text_raw[:500]}\n")

    # tokenize text after all preprocessing steps
    text = word_tokenize(text_raw)
    print_red("Text tokenized.")
    print(f"First 100 tokens:\n {text[:100]}\n")

    # stopwords removal
    if stopwords_removal:
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        print_red("Stopwords removed.")
        print(f"First 100 tokens after stopwords removal:\n {text[:100]}\n")

    # lemmatization or stemming (not both)
    if lemma:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
        print_red("Text lemmatized.")
        print(f"First 100 tokens after lemmatization:\n {text[:100]}\n")

    elif stem:
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        print_red("Text stemmed.")
        print(f"First 100 tokens after stemming:\n {text[:100]}\n")

    # STATISTICS SECTION
    # token count and unique tokens
    token_count = len(text)
    unique_tokens = len(set(text))
    print(f"Total number of tokens: {token_count}")
    print(f"Total number of unique tokens: {unique_tokens}")

    # token frequency count
    token_freq = Counter(text)
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

    # print first 10 and last 10 tokens
    print("\nFirst 10 tokens by frequency:")
    for token, freq in sorted_tokens[:10]:
        print(f"{token}: {freq}")

    print("\nLast 10 tokens by frequency:")
    for token, freq in sorted_tokens[-10:]:
        print(f"{token}: {freq}")

    # print all tokens if list_all is True
    if list_all:
        print("\nAll tokens by frequency:")
        for token, freq in sorted_tokens:
            print(f"{token}: {freq}")

    # MATPLOTLIB SECTION
    # visualization using Matplotlib
    ranks = range(1, len(sorted_tokens) + 1)
    frequencies = [freq for token, freq in sorted_tokens]

    plt.figure(figsize=(12, 8))

    # Plot 1: Rank vs Frequency
    plt.subplot(2, 2, 1)
    plt.plot(ranks, frequencies, marker='o')
    plt.title('Rank vs Frequency')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')

    # Plot 2: Log-Scale X
    plt.subplot(2, 2, 2)
    plt.plot(ranks, frequencies, marker='o')
    plt.xscale('log')
    plt.title('Log-Scale X: Rank vs Frequency')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency')

    # Plot 3: Log-Scale Y
    plt.subplot(2, 2, 3)
    plt.plot(ranks, frequencies, marker='o')
    plt.yscale('log')
    plt.title('Log-Scale Y: Rank vs Frequency')
    plt.xlabel('Rank')
    plt.ylabel('Frequency (log scale)')

    # Plot 4: Log-Scale Both
    plt.subplot(2, 2, 4)
    plt.plot(ranks, frequencies, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Log-Scale Both: Rank vs Frequency')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')

    plt.tight_layout()
    plt.show()

# dunder name
if __name__ == "__main__":
    main()
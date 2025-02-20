# main_menu.py

def print_menu():
    print("Second Homework (2/18/25) Menu:")
    print("Instructions: For '10. x', enter 10 on the command line and press enter. Same for others.  If it is a decimal such as '30.2 x', enter 30.2 and press enter.")
    print("Type 'quit' and press enter to quit.")
    print()

    # Basic Information
    print("#### Basic Information ####")
    print("1 Assignment Overview")
    print("2 Pipeline (all preprocessing steps with stemming)")
    print("3 Pipeline (all preprocessing steps with lemmatization)")
    print("4 Naive Bayes Pipeline (steps with stemming")
    print("4.5 Bag of Words Pipeline (steps with stemming)")
    print("6 Full Pipeline (ALL steps with stemming; run each step individually with diagnostics for easier analysis)")
    print()

    # Dataset
    print("#### Dataset ####")
    print("'10' Dataset Overview")
    print("'11' Load Dataset")
    print("'11.1' Diagnostics on Loaded Dataset")
    print("'12' Remove Punctuation")
    print("'12.1' Diagnostics on Removed Punctuation")
    print("'13' Lowercase")
    print("'13.1' Diagnostics on Lowercasing")
    print("'14' Tokenization")
    print("'14.1' Diagnostics on Tokenization")
    print("'15' Stopword Removal")
    print("'15.1' Diagnostics on Removal of Stopwords")
    print("'16' Stemming")
    print("'16.1' Diagnostics on Stemming")
    print("'17' Lemmatization")
    print("'17.1' Diagnostics on Lemmatization")
    print()

    # Bag of Words
    print("#### Bag-of-Words")
    print("20 Bag-of-Words Overview")
    print("20.1 Bag-of-Words, Unified and Per Author")
    print("21 Diagnostics on Bag-of-Words (per document)")
    print("22 Diagnostics on Bag-of-Words (author HG Wells)")
    print("23 Diagnostics on Bag-of-Words (author Shakespeare)")
    print("24 Diagnostics on Bag-of-Words (both authors)")
    print("25 Bag of Words Statistics")
    print("26 Unified Bag of Words Structure")
    print()

    # Naive Bayes
    print("#### Naive Bayes ####")
    print("30 Naive Bayes Overview")
    print("31 Naive Bayes Analysis (count)")
    print("32 Naive Bayes Analysis (binary)")
    print("33 Naive Bayes Analysis (tf-idf)")
    print()

    # Topic Modeling
    print("#### Topic Modeling ####")
    print("40 Topic Modeling Overview")
    print("41 Run Topic Modeling with Diagnostics")
    print()

    # Experimentation
    #print("#### Experimentation ####")
    #print("50 Experimentation Overview")
    #print("51 x")
    #print()

    # debug menu
    #print("#### DEBUG ####")
    #print("100. debug overview")
    #print("101. Compare lists")
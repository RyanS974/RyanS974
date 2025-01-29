# README.md
Ryan Smith
1/28/25
Homework 0
ARI 525 NLP

This is the readme file for the first homework assignment due 1/28/25.

# Files
The code files are all in this same directory.

- README.md, this file.
- HOMEWORK.md, the homework report.
- main-c.py, the main python file to run.
- pg1727.txt, the Project Gutenberg text file of the Odyssey

There are also some image files which are related to the report, and are included in the HOMEWORK.md file.

# Command-Line Options
Here are the command-line options:

- no_punctuation (additional option)
- expand_contractions (additional option; from 3rd party library 'contractions', 'pip3 install contractions')
- lowercase
- stopwords_removal
- stem
- lemma
- list_all (This will list all tokens sorted by frequency, most first (descending). The default stats are of just 10 of the most and 10 of the least.)

You can use just stem or lemma, but not both, since they don't work well together.  If you pass both options on the command-line, only one will work.  It checks for lemma first, so if you pass them both, only lemma will be processed.

# Beginning Diagnostics
Each option selected will print out brief diagnostic information, such as 500 characters of the raw string, or 100 tokens.

# Statistics
At the end of the file the top 10 tokens by frequency are listed, along with the last 10 tokens.  Before this, the total number of tokens is listed.  With all but lemmatization passed, there are ~59,000 tokens, and ~6,000 unique tokens.  If stopwords are not removed, the total token count is ~132,000.  If we just use --stem or --lemma we get ~150,000 total tokens.  Other combinations can produce different numbers also.

# Note on Lemmatization
Lemmatization produces minimal results, but it does produce some.  For example, Laws to Law, and simlar words.  This appears to be normal.  Stemming produces much more changing.  Technically, from my understanding, lemmatization is more advanced, with stemming simply removing prefixes and suffixes.  Lemmatization uses a dictionary approach to be more precise.

# Visualizations with Matplotlib
I have four main plots:

- normal
- log scale x
- log sclae y
- log scale both

The main concepts of these are the the log scale produces a pull in the direction of the scale, which is clearly visible.  It does help visualize the differences more, instead of major drops or flat lines of sorts in the plots without the scale, which is also clearly visible.
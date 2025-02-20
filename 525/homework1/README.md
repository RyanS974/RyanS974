# README.md
Ryan Smith
2/18/25
Homework 1
ARI 525 NLP

# Overview

This is the readme file for the second  homework assignment due on 2/18/25.  The main.py app is what is to be executed, which includes a menu based interface to go through the steps of the homework.

# Files
The code files are all in this same directory and listed below, along with all the related files of this assignment.

- README.md, This file.
- REPORT.md, The homework report.
- main.py, The main program for the assignment which contains the menu system. Run this.
- various helper files, such as bag_of_words.py, topic_modeling.py, datashare.py, naive_bayes.py.

# Menu

When you run main.py with ./main.py (this is how I run it on Mac), you will be presented with a menu system.  After results are displayed, or a response is displayed, you will be asked to press enter to continue.  It will then display the main menu again.  That is the basic interface.

Enter numbers such as:

10 (then press enter, don't include a period or quotes if it is not a decimal)
30.2 (then press enter, with a decimal needed in this case)

# Pipelines

If you want to run everything with stemming, this is the full pipeline in the menu:

6 Full Pipeline (ALL steps with stemming; run each step individually with diagnostics for easier analysis)

For easier analysis of what is happening, you can run them all individually with various diagnostic displays.

# Dataset

The dataset is of two authors from Project Gutenberg, HG Wells and Shakespeare.  The files are located in the directories of their names from the root directory.  They have sub-directories of the text file name such as pg###, then another sub-directory of either 'scenes' or 'chapters'.  That is where the documents are.  There are about 115 documents per author.  I used various shell scripts to split them, and I manually edited some of them.  The smallest and largest were removed, to achieve a fairly balanced document set.

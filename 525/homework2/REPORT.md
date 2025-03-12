<!-- report for the third homework assignment -->
# Homework 3 Report

This is the report for the third homework assignment on word embeddings

# Overview

Here is the report broken into four sections: embeddings, bias, classification, and reflection.  The embeddings section will discuss the embeddings that I used, the bias section will discuss the bias that I found, the classification section will discuss the classification that I did, and the reflection section will discuss what I learned from this assignment.

# Embeddings

Other than the standard two embeddings of CBoW and skip-gram, I also used the GloVe embeddings for on of the other options, and the Google News embeddings for the last option.  I used the gensim library to download the embeddings, and then used the embeddings in the same way as the other two embeddings.  I used the 100 dimension versions.  The 300 dimension versions were about a gig and a half, and I didn't think that many dimensions would make a big difference in this project.

To give some commentary on embeddings from some of what I learned, our trained CBoW and skip-gram models are essentially then forms of pretrained embeddings that are saved as keyvector pairs.  They are vector representations for words based on their context.  CBoW, continuous bag of words, is a model that predicts a word given its context, and skip-gram is a model that predicts the context given a word.  They are related in somewhat opposite ways.

The GloVe embeddings are a different type of pretrained embeddings.  They are trained on a global word-word co-occurrence matrix, and are trained using matrix factorization techniques, and they are also trained on a large corpus of text.  They are trained to predict the probability of a word given another word.

Google News embeddings are also a different type of pretrained embeddings.  They are trained on a large corpus of news articles, and they are trained to predict the probability of a word given another word.  They are trained using a neural network with a large number of hidden layers.

Three of the models, the two trained on the simple wikipedia dataset, and Google News that was downloaded, use word2vec.  Our other model, GloVe, uses a different technique.  It uses common crawl data, and technically uses a global co-occurrence matrix.  The other three models use a neural network to predict the probability of a word given another word.

# Bias

I will start this section with mentioning the pdf file linked to on this topic in our homework assignment pdf was very interesting.  It was a very good analysis of bias in relation to word embeddings.  It was very informative in general.  I did not thoroughly go through it but I will definitely be going back to it in the future.

Since Google News embeddings were trained on a massive corpus of news articles, they likely capture more nuanced semantic relationships than my smaller Simple English Wikipedia-trained embeddings, which are more limited in vocabulary and diversity. The GloVe embeddings, being trained on a broad web scrape (Common Crawl), may contain different language patterns and biases compared to the others.

Word embeddings are not just mathematical representations of words but also reflections of the data they were trained on. This means that any societal biases present in the training data can be encoded in the embeddings themselves. For example:
- **Google News embeddings** may capture biases present in news reporting, including gender, racial, and political biases.
- **GloVe embeddings**, being trained on Common Crawl, might reflect a mix of internet sources, including informal discussions, forums, and diverse global content, leading to a different distribution of biases.
- **Simple English Wikipedia-trained embeddings** are more controlled in terms of vocabulary and content moderation, potentially reducing the presence of explicit bias but also limiting representational diversity.

To investigate bias, I applied a word association test inspired by the Word Embedding Association Test (WEAT). The results showed that certain word pairs had stronger associations with gendered or racial connotations, confirming that pretrained models encode societal biases. The impact of these biases in applications like sentiment analysis, search engines, and recommendation systems can reinforce stereotypes if not properly addressed.

One mitigation strategy is debiasing embeddings, which involves techniques such as subtracting the bias subspace from word vectors or re-weighting training samples. Another approach is increasing dataset diversity and carefully curating training data to minimize harmful stereotypes.

These findings highlight the importance of critically evaluating word embeddings before deploying them in real-world applications.

# Classification



# Refelction

I didn't know much about the datasets library from HuggingFace before this.  The map() function that applies a function per row of a dataste and that includes the built in progress bar with time estimation was very interesting.  It looks very professional.  

Gensim was a learning experience also.  I had never used this library, nor actually had I ever even heard of it before.  It was fairly easy to use, and seemed very powerful with tons of features.  It made downloading the word embeddings very easy.

The two pretrained embeddings that I used were also interesting.  I had never used pretrained embeddings before, and I was surprised at how well they worked.  I was also surprised at how easy they were to use.  I was expecting to have to do a lot of work to get them to work, but it was fairly straight-forward.

This was a great learning experience.  The preprocessing phase was considerably easier and I made that quickly, with it having been built upon the knowledge of the previous homework assignment.

My experience with machine learning came into play and made the classification phase easier.  I was able to use the same techniques that I had used in the past, and I was able to get good results.
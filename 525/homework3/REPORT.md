# Report

This is the report for the 4th homework assignment.

# 1. Dataset

## The Task I am Working On

The task I've picked for this homework is a binary text classification using the SMS Spam Detection dataset from HuggingFace.  The task is of identifying whether an SMS message is spam or not.  If it is not spam it is considered legitimate ham.  This is a major NLP task that is widely used and very useful, that of identifying spam messages.  For the Baselines section I will use a Logistic Regression model with a Bag of Words representation and TF-IDF representation.  Related to this will be the random and majority class baselines.  The fine-tuned models will be DistilBERT and T5.  I will also use the ExaOne 3.5 and Granite 3.2 models for zero-shot classification, which will use the requests library to access a local Ollama server to run these LLMs.

## How the Data was Originally Collected

The corpus was collected by Tiago Almeida and Jose Hidalgo, and is of a paper of theirs titled "Contributions to the Study of SMS Spam Filtering: New Collection and Results".  The corpus has been collected from free or free for research sources at the Web and compiled from multiple sources:
1. 425 SMS spam messages extracted manually from the Grumbletext Web site.
2. 450 SMS ham messages collected from Caroline Tag's PhD Thesis.
3. A subset of 3,375 SMS ham messages from the NUS SMS Corpus (NSC) (volunteers from a Singapore University helped).
4. 1,002 SMS ham messages and 322 spam messages extracted from the SMS Spam Corpus v.0.1 Big.

Those are the main four sources of the data.  The dataset is in English and contains real SMS messages from actual users, making it representative of real-world communications.  It is a widely used benchmark dataset in SMS spam detection research.

## The Inputs / Outputs

The inputs are SMS text messages in English, real and non-encoded.  The messages vary in length and content, from short personal communications to promotional messages.  The outputs are binary classification with two classes:
- **Ham**: Legitimate messages (labeled as 'ham')
- **Spam**: Unsolicited commercial or fraudulent messages (labeled as 'spam')

## The Evaluation Metrics

The evaluation metrics for this binary classification task are:
- **Accuracy**: Provides an overall measure of correct predictions but can be misleading due to class imbalance.
- **Precision**: In the spam classification example, precision measures the fraction of emails classified as spam that were actually spam. This is important to ensure that legitimate messages aren't incorrectly filtered.
- **Recall**: In the spam classification example, recall measures the fraction of spam emails that were correctly classified as spam. This tells us how effective our model is at catching all spam.
- **F1-Score**: Since the imbalanced nature of the dataset (more ham than spam), F1-score is particularly relevant.  The F1 score is the harmonic mean of precision and recall. This metric somewhat balances the importance of precision and recall, and is preferred to accuracy for class-imbalanced datasets.

These are the standard four machine learning metrics for binary classification.  The F1-score is particularly relevant for this dataset because it is imbalanced, with 86.6% ham messages and only 13.4% spam messages.  The F1-score is a better metric than accuracy in this case because of the class imbalance.

## Dataset Statistics

| Split | Total Examples | Ham (legitimate) | Spam |
|-------|----------------|------------------|------|
| Total | 5,574          | 4,827 (86.6%)    | 747 (13.4%) |
| Train (80%) | 4,459    | 3,861 (86.6%)    | 598 (13.4%) |
| Test (20%)  | 1,115    | 966 (86.5%)      | 149 (13.5%) |

The dataset does not come with predefined train/test splits, so for this assignment, I've used an 80/20 random split which is requested in the pdf.

# 2. Fine-tuned models

DistilBERT and T5 are the fine-tuning models.

DistilBERT has 66 million parameters.  It is a smaller, faster, cheaper, and lighter version of BERT.  It is a transformer-based model that uses the same architecture as BERT but with fewer parameters.  It was pre-trained on the same dataset as BERT, which is the BookCorpus and English Wikipedia.

The fine-tuning approach used was:
- Hyperparameters: learning rate of 2e-5, batch size of 16, and 3 epochs.
- Training time: 
- Hardware used: M4 Macmini with 24gb of RAM.
- Optimization technique: AdamW optimizer with weight decay.
- Data preprocessing: AutoTokenizer from HuggingFace was used to tokenize the data.  The data was also padded and truncated to a maximum length of 128 tokens.

T5 has 220 million parameters.  It is a transformer-based model that uses the same architecture as BERT but with a different pre-training objective.  It was pre-trained on the C4 dataset, which is a large-scale web corpus.

The fine-tuning approach used was:
- Hyperparameters: learning rate of 2e-5, batch size of 16, and 3 epochs. (same as DistilBERT)
- Training time: 
- Hardware used: M4 Macmini with 24gb of RAM.
- Optimization technique: AdamW optimizer with weight decay. (same as DistilBERT)
- Data preprocessing: AutoTokenizer from HuggingFace was used to tokenize the data.  The data was also padded and truncated to a maximum length of 128 tokens. (same as DistilBERT)



# 3. Zero-shot classification

For the zero-shot classification I used exaone 3.5 from LG and granite 3.2 from IBM.  These were both run on my machine, an M4 Macmini with 24gb of RAM.  

## Exaone 3.5:

- Exaone 3.5 is developed by LG AI Research. It is a multimodal AI model.
- It is designed to handle various tasks, including generating images, videos, and text.
- Exaone 3.5 is intended to be a foundation model for AI applications across various industries.
- Based on information found, it is designed for advanced reasoning, and creative content creation.
- It is designed to be able to understand complex context.
- More specific information on the training data, parameters, and training hardware is limited to the public (if it is available, I couldn't find it).
- It is built with a focus on multimodality.

- The specific version I used was 7.8 billion parameters.  It is a smaller version of the model, which is designed to be more efficient and faster than the larger versions.
- This version is 4.8gb in size and is designed to be run on a single GPU.

## Granite 3.2:

- Granite 3.2 is a family of large language models (LLMs) developed by IBM.
- It builds upon the previous Granite 3.1 model.
- It comes in various sizes, including 2B and 8B parameter versions.
- Granite 3.2 is fine-tuned with a combination of open-source and IBM-generated synthetic data, emphasizing reasoning capabilities.
- It supports 12 languages, with the ability to be fine-tuned for others.
- It is designed for enterprise level applications.
- The models include a focus on controllable reasoning.
- The training data includes a focus on coding, and general language understanding.
- Like Exaone 3.5, the exact hardware configurations used for pretraining are not widely publicized, but given the model size, it is safe to say they are extensive GPU clusters.

- The specific version I used was 8 billion parameters.
- This version is 4.9gb in size and is designed to be run on a single GPU.

## Talbe of more zero-shot model information

Here is some more detailed information on the zero shot models by LG and IBM.  This was the most detailed information I could find on them.

| Metric              | LG Exaone 3.5 (32B)       | IBM Granite 3.2 (8B Instruct / 2B Instruct) |
|---------------------|---------------------------|---------------------------------------------|
| Model Size          | 32 Billion                | 8 Billion / 2 Billion                       |
| Training Tokens     | 6.5 Trillion              | 12 Trillion                                 |
| FLOPs               | 1.25 × 10^24              | Not Available                               |
| Hardware (Training) | NVIDIA H100 GPUs          | NVIDIA H100 GPUs                            |

Also, I could not find much training time information, but I found that a similar Granite model was pretrained in 30 hours, with this model being just a billion parameter one.  So an estimate is 8 times that, or 240 hours.  This is just a guess though, and the actual time could be more or less.

I mainly use Granite 3.2 from IBM and Exaone 3.5 from LG.  I have quite a few downloaded, but those are the main ones I use.  They are about 5gb each, and about 8 billion parameters each.  If I want more accuracy, I usually use Microsoft's Phi4 model that is 9.1gb.  I also will use qwen2.5-coder for code specific tasks, with there being a considerable amount of versions.  The smaller versions work well as auto-complete models in IDEs where they don't take up too many resources.  I use those often in VS Code with the Continue extension that supports Ollama.  My machine will not run anything over about 9gb, model wise, with my M4 mac mini having 24gb.  The over 9gb models I have tested are extremely slow and take up a consderable amount more of memory, and don't appear to be just a linear increase in memory usage.

# 4. Baselines

The baseline models are of a Logistic Regression model with a Bag of Words representation and TF-IDF representation.  The random and majority class baselines are also included.  The input features are TF-IDF, which is essentially a weighted Bag of Words, from my understanding.  I first get the Bag of Words representation, and then I get the TF-IDF representation, although they are not really based on each other directly.  I was somewhat confused in the pdf phrasing, as it spoke of a BoW baseline model with an input feature method.  I chose TF-IDF, but the Logistic Regression classifier is only working from the TF-IDF, which I believe is ok.  From my understanding, TF-IDF is considered somewhat of a weighted BoW.  The Logistic Regression model is a simple linear model that is used for binary classification.  It is a good baseline model to use for text classification tasks.

The random baseline is a model that randomly predicts the class of the input.  This is a very simple model that is used to compare against the other models.  The majority class baseline is a model that always predicts the majority class, which in this case is ham.  This is also a very simple model that is used to compare against the other models.



# 5. Results

Here is the main results table:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DistilBERT | 0.9910 | 0.9664 | 0.9664 | 0.9664 |
| T5 | 0.9839 | 0.9517 | 0.9262 | 0.9388 |
| ExaOne 3.5 | 0.8422 | 0.4516 | 0.8456 | 0.5888 |
| Granite 3.2 | 0.8924 | 0.5668 | 0.8255 | 0.6721 |
| LR TF-IDF (BoW) Baseline | 0.9776 | 1.0000 | 0.8322 | 0.9084 |
| Random Baseline | 0.7682 | 0.1348 | 0.1356 | 0.1350 |
| Majority/Target-Class Baseline | 0.8664 | 0.0000 | 0.0000 | 0.0000 |

LR above stands for Logistic Regression in the table.

## Fine-tuned Models

Results of the two fine-tuned models.

### DistilBERT

The high accuracy (0.9910) compared to the slightly lower precision/recall/F1 (0.9664) indicates that the model performs better overall than on just the positive class.  This makes sense if the dataset is imbalanced with more non-spam than spam messages, which is true in our case.  When the precision and recall are the same, the F1 score is also the same.  The model predicted very well.

### T5

The T5 model performed slightly worse than DistilBERT, with an accuracy of 0.9839 and a precision of 0.9517.  The recall was also lower at 0.9262, which indicates that the model is not as good at identifying spam messages as DistilBERT.  The F1 score was also lower at 0.9388, which indicates that the model is not as good at identifying spam messages as DistilBERT.

### DistilBERT vs T5

The DistilBERT model performed better than the T5 model in all metrics.  This is likely due to the fact that DistilBERT is a smaller model and is more efficient at identifying spam messages.  The T5 model is larger and more complex, which may have made it less effective at identifying spam messages.

## Zero-shot Classification

For the zero-shot classification, I go through 50 random samples per prompt type, of which there are 5.  This is for both exaone and granite.  This is to find the best prompt.  

```python
# List of prompts to try
    prompts = [
        "Is the following SMS message spam? Respond with 1 for spam, 0 for not spam: '{text}'",
        "Classify the following SMS as spam (1) or not spam (0): '{text}'",
        "Analyze this message and determine if it's spam. Reply with 1 for spam or 0 for not spam: '{text}'",
        "Is this SMS message legitimate or spam? Answer 0 for legitimate, 1 for spam: '{text}'",
        "This is an SMS message: '{text}' Is this spam? Answer with a single digit: 1 for yes, 0 for no."
    ]
```

Above is the actual prompts from the code.  For exaone, the fifth one performed the best.  I then went through all 1115 samples of the test set for the scores.  This didn't take that long.  Possibly about an hour or so.  It was the same for granite also.

### Exaone 3.5

The Exaone 3.5 model performed well, with an accuracy of 0.8422.  The precision was 0.4516, which indicates that the model is not as good at identifying spam messages as the fine-tuned models.  The recall was 0.8456, which indicates that the model is good at identifying spam messages, but not as good as the fine-tuned models.  The F1 score was 0.5888, which indicates that the model is not as good at identifying spam messages as the fine-tuned models.  This was extremely interesting using Ollama LLMs for this type of project.  I think the model did fairly well.  Not nearly as good as the fine-tuned models, but with it being a zero-shot method it did ok I think.

### Granite 3.2

On why Granite performed better: It is more technically trained, I believe.  The model size is smaller, but the training tokens is larger, for one.  Also, I believe the LG Exaone model is focused more on being multi-lingual, which could be a factor.  It could be a trade off of sorts there, to be able to better support more languages.  Granite does have the Granite-code line of models, which could be a related factor also, making it more technical in nature.  I was expecting a closer result.  Granite didn't perform a ton better, but it did give better results.  Both of these models, Exaone and Granite, performed fairly well though for zero-shot.

The actual Granite scores were:
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Granite 3.2 | 0.8924 | 0.5668 | 0.8255 | 0.6721 |

The Granite 3.2 model performed better than the Exaone 3.5 model, with an accuracy of 0.8924.  The precision was 0.5668, which indicates that the model is not as good at identifying spam messages as the fine-tuned models.  The recall was 0.8255, which indicates that the model is good at identifying spam messages, but not as good as the fine-tuned models.  The F1 score was 0.6721, which indicates that the model is not as good at identifying spam messages as the fine-tuned models.

## Baselines

For the Logistic Regression baseline section, the random and majority performed as expected.  

# Reflection

As expected, DistilBERT performed the best.  I knew it would be that or T5.  DistilBERT, despite being a less technically advanced model than T5 in many ways, performed better.  I attributed this to the simple nature of the project and sms messages worked well with the model.  It has a good trade-off of performance and efficiency, which made it a good choice here.  It was also both smaller and faster technically than T5.  The more advanced aspects of T5 appeared to not be needed and hindered this more simple binary text classification task.

I think this assignment was the most interesting one so far in the course.  I found working with Ollama through the requests library very interesting.  I had never actually used requests like this, and from my research, it was the possible best method, bypassing the actual Ollama api calls.  I was surprised it worked as well as it did.  It ran fairly fast also.  I think they both took a little over an hour each to complete.  Granite took longer for some reason, although it is only technically .1 gb larger of an LLM.  I am not sure why on that aspect.  

As for why the Exaone worked best with the fifth of five prompt test prompts, and Granite the second, I am not sure also.  I believe Exaone as more general purpose training, which might have been a factor in that.  The fifth prompt is more technical sounding, and technical in general I believe, which might have made it easier for Exaone to understand.  It begins with saying 'This is an SMS message...' which might have helped Exaone understand, whereas Granite did not need it and worked better with the second prompt which was "Classify the following SMS as spam (1) or not spam (0): '{text}'".  Prompt five was "This is an SMS message: '{text}' Is this spam? Answer with a single digit: 1 for yes, 0 for no."  Also, as I mentioned Granite took a decent amount of more time to complete the 1115 samples classification.  Exaone is 7.8 billion parameters and 4.8gb, Granite is 8 billion parameters and 4.9gb, but the time increase was not linear.  Again, I attribute it to the training of the model and the more technical nature of it in general.  I could be wrong, but that is my view.

Baseline wise, I was somewhat familiar with the logistic regression model and process, so it went fairly well.  It performed fairly well also.  This assignment took somewhat less time to complete than the prior one, although still a decent amount of time overall, and was also somewhat easier for me, but was still overall fairly challenging, and I learned much completing it.  I am looking forward to working on the Gradio app homework next, as I already have a good idea of what I am going to do with that, and how to implement it.
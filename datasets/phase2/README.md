# How to participate

# Candidate Evaluation Competition

## Project Overview
Welcome to the Candidate Evaluation Competition! This competition involves building a multi-label classification model to predict:

- **Hire Decision**: Whether a candidate will be hired (`Yes`, `No`, or `Interview`).
- **Pay Category**: The expected salary for the candidate (`None`, `125`, or `150`).

Using a labeled dataset of candidate profiles, participants will train models locally, compute their model performance on an unlabeled dataset, and submit their predicted labels in a text file to the CodaLab page.  You can use any classification model you want, such as Random Forest, etc.

## Dataset Description
The provided dataset includes features like:

- **Skills**: Technical skills (e.g., Python, SQL, Java).
- **Experience**: Years of experience.
- **Grades**: Candidate’s average grade.
- **Projects**: Number of completed projects.
- **Extracurriculars**: Indicator for involvement in extracurricular activities.

### Data Files Locations
The datasets are provided on my CodaLab page and GitHub.

- training_set_labeled.csv: Contains features and labels for each candidate. 400 datapoints for training.
- validation_set_unlabeled.csv: Contains just the features. 300 datapoints for validation.
- test_set_unlabeled.csv: Contains just the features. 300 datapoints for testing.

https://github.com/RyanS974/RyanS974/tree/main/datasets/phase2

### Example Code

At the same github address is two .ipynb files that is of the code for everything from the loading of the dataframes, running the classification in the training phase, getting the metric f1-score in the validation phase, and various other things.

The example workflow would be:
- train the model with training_set_labeled.csv
- validate the model with hyperparameter tuning and metrics using validation_set_unlabeled.csv
- test your model with test_set_unlabeled.csv (this is what your submission file preds.txt is based on)

**The files are named:**

- submission_instructions.ipynb
- combine.py also has some code for loading and predicting. This will write a preds.txt file also that would need to be zipped before submission.  This is of the baseline model for CodaLabs and would need to be modified, such as hyperparameters modification, of the three that are in the code, or added hyperparameters.  You could modify this however you want also with a new classifier also.  This is a simple example using the training set and test set.  the validation set is not used.

## Submission Instructions
Make sure your labels from your validation results are in a zip file before submitting it to CodaLabs.  The file inside should be named preds.txt with just the labels.

## Evaluation
Participants will be ranked based on their `overall_score` (average of hire and pay f1 scores). This composite score reflects the model's performance across both prediction labels.

### Additional Notes
- You can use any model or ensemble method to maximize your scores.
- Hyperparameter tuning and feature engineering are encouraged.
- The goal is to achieve the best possible performance on both `hire` and `pay` labels.

# Terms and Conditions

## Dataset License
The dataset used in this competition is sourced from Kaggle and is licensed under the Apache License, Version 2.0. By participating in this competition, you agree to comply with the terms of this license.

Here is the dataset Kaggle url:

https://www.kaggle.com/datasets/tarekmuhammed/job-fair-candidates

### License Summary
- You are free to use, modify, and redistribute the dataset as permitted under the Apache License, Version 2.0.
- Proper attribution to the dataset source is required.

### Full License Text
For full details of the Apache 2.0 License, please refer to the following link: [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Questions and Support
If you need help or have questions, reach out on the competition forum or email me at rysmith _AT_ umich _DOT_ edu.

Good luck!

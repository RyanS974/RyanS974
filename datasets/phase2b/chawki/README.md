# Sentiment Analysis Competition - Starter Kit

Welcome to the **Sentiment Analysis** competition! This repository provides the starter kit for the competition.

### Dataset Information for the train_data.csv
- **Columns**:
  - `ID`: Unique identifier for each review.
  - `Reviewer Name`: Username of the reviewer (you can ignore this during training)
  - `Review Text`: The text of the review, which you will use to train your sentiment analysis model
  - `Source`: Link to the source of the review (you can ignore this).
  - `Date of Collection`: The date when the review was collected (you can ignore this).
  - `Annotator_1`: Sentiment label assigned by annotator 1
  - `Annotator_2`: Sentiment label assigned by annotator 2
  - `Ground_Truth`: The final sentiment label for each review (used for training your model).


## Dataset Exploration

You can explore the dataset using the `data_exploration.ipynb` notebook. This notebook contains some basic data exploration steps, including:

- Loading the dataset
- Checking for missing values
- Visualizing the distribution of sentiment labels

## Training a Model

The `train_model.py` script provides a basic template for training a model. The script uses a Naive Bayes classifier with `CountVectorizer` for text processing. You can adjust the model or add your own improvements.

## Data Folder

The `data` folder contains the following files:
- `train_data.csv`
- `test_data.csv`

## License
This dataset is available under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
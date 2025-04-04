# dataset.py

# import huggingface datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset

app_data = {
    "dataset": {
        "name": "SMS Spam Detection",
        "description": "The SMS Spam Detection dataset contains 5,574 SMS messages, of which 4,827 are ham and 747 are spam.",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/ucirvine/sms_spam",
        "training_set": None,  # Training set of the dataset
        "test_set": None,  # Test set of the dataset
    },
    "fine_tune": {
        "distilbert": {
            "model": None,  # Fine-tuned DistilBERT model
            "results": None  # Evaluation metrics for DistilBERT
        },
        "t5": {
            "model": None,  # Fine-tuned T5 model
            "results": None  # Evaluation metrics for T5
        },
        "comparison_results": None  # Comparison of DistilBERT and T5 results
    },
    "zero_shot": {
        "exaone": {
            "results": None  # Evaluation metrics for Exaone3.5
        },
        "granite": {
            "results": None  # Evaluation metrics for Granite3.2
        },
        "comparison_results": None  # Comparison of Exaone3.5 and Granite3.2 results
    },
    "baselines": {
        "bow_representation": None,  # Bag-of-Words representation of the dataset
        "tfidf_representation": None,  # TF-IDF representation of the dataset
        "bow_model": None,  # TF-IDF input features (model is logistic regression)
        "bow_results": None,  # Evaluation metrics for the model
        "random_baseline": None,  # Evaluation metrics for the random baseline
        "majority_class_baseline": None  # Evaluation metrics for the majority class baseline
    },
    "conclusion": {
        "summary": None  # Final summary and insights from the project
    }
}

def loaddataset():
    """
    Load the SMS Spam Detection dataset from HuggingFace.
    The dataset contains 5,574 SMS messages, of which 4,827 are ham and 747 are spam.
    """
    # load dataset
    raw_dataset = load_dataset("ucirvine/sms_spam", split="train")  # Load the full dataset as a single split

    # Convert to pandas DataFrame for splitting
    df = raw_dataset.to_pandas()

    # Perform train-test split (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Convert back to Hugging Face Dataset
    app_data["dataset"]["training_set"] = Dataset.from_pandas(train_df)
    app_data["dataset"]["test_set"] = Dataset.from_pandas(test_df)

    print("Dataset loaded successfully.")

    # Return as a dictionary with train and test splits
    return app_data

def verifydataset():
    """
    Verify the dataset by checking the number of samples and the number of classes.
    """
    # check number of samples
    print("Number of samples in train set:", len(app_data["dataset"]["training_set"]))
    print("Number of samples in test set:", len(app_data["dataset"]["test_set"]))
    
    # check number of classes of train set
    print("Number of classes in train:", len(set(app_data["dataset"]["training_set"]["label"])))

    # check number of classes of test set
    print("Number of classes in test:", len(set(app_data["dataset"]["test_set"]["label"])))
    
    # calculate and display class distribution in train set
    train_labels = app_data["dataset"]["training_set"]["label"]
    class_distribution = {label: train_labels.count(label) for label in set(train_labels)}
    print("Class distribution in train set:", class_distribution)

    # calculate and display class distribution in test set
    test_labels = app_data["dataset"]["test_set"]["label"]
    class_distribution_test = {label: test_labels.count(label) for label in set(test_labels)}
    print("Class distribution in test set:", class_distribution_test)

    # check first 5 samples of train set
    print("First 5 samples train set:")
    print(app_data["dataset"]["training_set"][:5])

    # check first 5 samples of test set
    print("First 5 samples of test set:")
    print(app_data["dataset"]["test_set"][:5])
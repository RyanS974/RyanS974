import pandas as pd
import ssl
import certifi
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Set up SSL context to use certifi's certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Custom function to load CSV files with SSL context
def load_csv_with_ssl(url, column_names):
    with urllib.request.urlopen(url, context=ssl_context) as response:
        return pd.read_csv(response, names=column_names).drop(columns=["id"])

# Dataset URLs
train_dataset_url = 'https://raw.githubusercontent.com/RyanS974/RyanS974/main/datasets/phase2/training_set_labeled.csv'
test_dataset_url = 'https://raw.githubusercontent.com/RyanS974/RyanS974/main/datasets/phase2/test_set_unlabeled.csv'

# Column names
column_names = ["id", "skills", "exp", "grades", "projects", "extra", "offer", "hire", "pay"]
column_names2 = ["id", "skills", "exp", "grades", "projects", "extra", "offer"]

# Load the datasets
train_data = load_csv_with_ssl(train_dataset_url, column_names)
test_data = load_csv_with_ssl(test_dataset_url, column_names2)

# Separate features and labels for the training dataset
X_train = train_data.drop(columns=["hire", "pay"])  # Features for training
y_train = train_data[["hire", "pay"]]               # Labels for training

# Validation features without dropping any additional columns
X_test = test_data  # Features for validation

# Print shapes to verify the split
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape)

# Initialize CountVectorizer for skills column
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(";"))

# Set up the preprocessor with CountVectorizer for the skills column
preprocessor = ColumnTransformer(
    transformers=[
        ("skills", vectorizer, "skills")
    ],
    remainder="passthrough"  # Keeps the other (numerical) columns as they are
)

# Create a pipeline to combine preprocessing and model training
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=300, max_depth=None, max_features=1.0, random_state=42)))
])

# Train the model on the training set
pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_test_pred = pipeline.predict(X_test)

# Write predictions to a file in "value,value" format
with open("preds.txt", "w") as file:
    for prediction in y_test_pred:
        file.write(",".join(map(str, prediction)) + "\n")


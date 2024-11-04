import pandas as pd

train_dataset_url = 'https://raw.githubusercontent.com/RyanS974/RyanS974/main/datasets/phase2/training_set_labeled.csv'
val_dataset_url = 'https://raw.githubusercontent.com/RyanS974/RyanS974/main/datasets/phase2/validation_set_unlabeled.csv'

# column names
column_names = ["id", "skills", "exp", "grades", "projects", "extra", "offer", "hire", "pay"]
column_names2 = ["id", "skills", "exp", "grades", "projects", "extra", "offer"]

# Load the datasets
train_data = pd.read_csv(train_dataset_url, names=column_names).drop(columns=["id"])
val_data = pd.read_csv(val_dataset_url, names=column_names2).drop(columns=["id"])

# Separate features and labels for the training dataset
X_train = train_data.drop(columns=["hire", "pay"])  # Features for training
y_train = train_data[["hire", "pay"]]               # Labels for training

# Validation features without dropping any additional columns
X_val = val_data  # Features for validation

# Print shapes to verify the split
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape)

# Import necessary libraries for encoding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

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
y_val_pred = pipeline.predict(X_val)

# Write predictions to a file in "value,value" format
with open("submission.txt", "w") as file:
    for prediction in y_val_pred:
        file.write(",".join(map(str, prediction)) + "\n")


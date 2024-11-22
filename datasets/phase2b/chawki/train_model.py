from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load  data
train_data = pd.read_csv('data/train_data.csv')

# Features and labels
X_train = train_data['Review Text']
y_train = train_data['Ground_Truth']

# Create a simple model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)


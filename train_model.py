import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("data/pnec_dataset.csv")

# Get questions and categories
questions = data["question"]
categories = data["category"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, categories)


# Function to predict category
def predict_category(user_question):

    question_vector = vectorizer.transform([user_question])

    prediction = model.predict(question_vector)

    return prediction[0]


# Test the model when this file is run directly
if __name__ == "__main__":

    print("AI Model trained successfully!")

    question = "Does PNEC offer BS Computer Science?"

    category = predict_category(question)

    print("Predicted Category:", category)
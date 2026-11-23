
import pandas as pd
from train_model import predict_category

# Load dataset
data = pd.read_csv("data/pnec_dataset.csv")


# Words that are not useful for searching
stop_words = [
    "what", "is", "the", "a", "an",
    "can", "i", "does", "do", "how",
    "to", "of", "for", "and", "me",
    "tell", "please", "about", "at", "pnec"
]


# Clean the question
def clean_words(sentence):

    sentence = sentence.lower()

    for mark in "?,.!":
        sentence = sentence.replace(mark, "")

    words = sentence.split()

    useful_words = []

    for word in words:
        if word not in stop_words:
            useful_words.append(word)

    return useful_words


# Find the best answer
def find_answer(user_question):

    # Step 1: Check exact question match
    entered_question = user_question.lower().strip()

    for mark in "?,.!":
        entered_question = entered_question.replace(mark, "")

    for index, row in data.iterrows():

        stored_question = str(row["question"]).lower().strip()

        for mark in "?,.!":
            stored_question = stored_question.replace(mark, "")

        if entered_question == stored_question:
            return row["answer"]

    # Step 2: Predict category using AI
    category = predict_category(user_question)

    print("Predicted Category:", category)

    # Step 3: Clean user's question
    user_words = clean_words(user_question)

    if len(user_words) == 0:
        return "Sorry, I could not find an answer to your question."

    # Step 4: Search the complete dataset
    best_score = 0
    best_answer = None

    for index, row in data.iterrows():

        stored_words = clean_words(str(row["question"]))

        matching_words = 0

        for word in user_words:
            if word in stored_words:
                matching_words += 1

        # Calculate Jaccard similarity
        total_words = len(set(user_words + stored_words))

        if total_words > 0:
            score = matching_words / total_words
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_answer = row

    # Step 5: Return answer only if match is good enough
    if best_answer is not None and best_score >= 0.30:
        return best_answer["answer"]

    return "Sorry, I could not find an answer to your question."

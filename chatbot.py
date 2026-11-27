
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from train_model import predict_category


# Load dataset
data = pd.read_csv("data/pnec_dataset.csv")


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Prepare questions
questions = data["question"].fillna("").astype(str).tolist()
cleaned_questions = [clean_text(q) for q in questions]


# Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(cleaned_questions)


# Identify the category of a question
def get_category(question):

    q = clean_text(question)

    # Computer Science
    if any(word in q for word in [
        "computer science", "cs department", "bscs",
        "programming", "data science", "cyber security"
    ]):
        return ["Computer Science"]

    # Departments
    if any(word in q for word in [
        "department", "departments", "hod", "faculty"
    ]):
        return ["Departments"]

    # Library
    if any(word in q for word in [
        "library", "book", "books", "printing"
    ]):
        return ["Library"]

    # Hostel
    if any(word in q for word in [
        "hostel", "accommodation", "dormitory"
    ]):
        return ["Student Life", "Admissions"]

    # Contact
    if any(word in q for word in [
        "phone", "number", "email", "contact", "registrar"
    ]):
        return ["Contact"]

    # Admissions
    if any(word in q for word in [
        "admission", "apply", "eligibility", "joining",
        "scholarship", "financial aid", "fee", "fees",
        "tuition", "net", "merit", "entry test",
        "matric", "intermediate", "percentage",
        "sat", "act", "nfaaf", "loan", "instalment",
        "installment"
]):
        return ["Admissions"]

    # Transport
    if any(word in q for word in [
        "transport", "bus", "route"
    ]):
        return ["Transport"]

    # General PNEC questions
    if any(phrase in q for phrase in [
        "what is pnec",
        "tell me about pnec",
        "about pnec",
        "what does pnec stand for",
        "where is pnec located",
        "when was pnec established"
    ]):
        return ["General"]

    return []


# Find the best answer
def find_answer(user_question):

    cleaned_input = clean_text(user_question)

    if not cleaned_input:
        return "Please enter a valid question."

    # Reject unsupported facility questions
    if any(word in cleaned_input for word in [
        "swimming pool", "pool"
    ]):
        return "Sorry, I do not have verified information about this facility at PNEC."

    # Exact question matching
    for index, question in enumerate(cleaned_questions):

        if cleaned_input == question:
            return data.iloc[index]["answer"]

    # Predict category using AI model
    try:
        print("Predicted Category:", predict_category(user_question))
    except Exception:
        pass

    # Identify relevant categories
    categories = get_category(user_question)

    if not categories:
        return "Sorry, I could not find an answer to your question."

    # Safe response for hostel availability
    if "hostel" in cleaned_input and any(word in cleaned_input for word in [
        "stay", "live", "available", "availability", "accommodation"
    ]):
        return "I do not have verified information about current hostel availability. Please contact PNEC Student Affairs."

    # Select questions from relevant categories only
    candidate_indices = data[
        data["category"].isin(categories)
    ].index.tolist()

    if not candidate_indices:
        return "Sorry, I could not find an answer to your question."

    # Calculate similarity
    user_vector = vectorizer.transform([cleaned_input])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors[candidate_indices]
    )[0]

    best_position = similarity_scores.argmax()
    best_index = candidate_indices[best_position]
    best_score = similarity_scores[best_position]

    print("Matched Question:", data.iloc[best_index]["question"])
    print("Similarity Score:", round(best_score, 3))
    print("Matched Category:", data.iloc[best_index]["category"])

    # Return answer only if similarity is sufficient
    if best_score >= 0.30:
        return data.iloc[best_index]["answer"]

    return "Sorry, I could not find an answer to your question."


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


# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(cleaned_questions)


# Identify relevant category
def get_category(question):

    q = clean_text(question)

    # Sports and student activities
    if any(word in q for word in [
        "sport", "sports", "gym", "gymnasium",
        "badminton", "cricket", "tennis", "squash",
        "basketball", "volleyball", "futsal",
        "snooker", "table tennis", "football",
        "competition", "society", "societies",
        "club", "clubs", "faculty advisor",
        "counselling", "counseling"
    ]):
        return ["Student Life"]

    # Hostel
    if any(word in q for word in [
        "hostel", "accommodation", "dormitory",
        "room allocation"
    ]):
        return ["Student Life", "Admissions"]

    # Contact
    if any(word in q for word in [
        "phone", "number", "email", "contact",
        "registrar", "address", "telephone"
    ]):
        return ["Contact"]

    # Library
    if any(word in q for word in [
        "library", "book", "books", "printing",
        "periodical", "thesis", "theses"
    ]):
        return ["Library"]

    # Transport
    if any(word in q for word in [
        "transport", "bus", "route", "pick up",
        "drop off"
    ]):
        return ["Transport"]

    # Computer Science
    if any(word in q for word in [
        "computer science", "cs department", "bscs",
        "programming", "data science", "cyber security",
        "artificial intelligence"
    ]):
        return ["Computer Science"]

    # Departments
    if any(word in q for word in [
        "department", "departments", "hod"
    ]):
        return ["Departments"]

    # Admissions and financial aid
    if any(word in q for word in [
        "admission", "apply", "eligibility", "joining",
        "scholarship", "financial aid", "fee", "fees",
        "tuition", "net", "merit", "entry test",
        "matric", "intermediate", "percentage",
        "sat", "act", "nfaaf", "loan",
        "instalment", "installment"
    ]):
        return ["Admissions"]

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


# Find answer
def find_answer(user_question):

    cleaned_input = clean_text(user_question)

    if not cleaned_input:
        return "Please enter a valid question."

    # Unknown facility protection
    if any(word in cleaned_input for word in [
        "swimming pool", "pool"
    ]):
        return "Sorry, I do not have verified information about this facility at PNEC."

    # Exact question matching
    for index, question in enumerate(cleaned_questions):

        if cleaned_input == question:
            return data.iloc[index]["answer"]

    # Detect category
    categories = get_category(user_question)

    if not categories:
        return "Sorry, I could not find an answer to your question."

    # Hostel availability protection
    if "hostel" in cleaned_input and any(word in cleaned_input for word in [
        "stay", "live", "available", "availability",
        "accommodation", "room allocation"
    ]):
        return (
            "I cannot confirm current hostel availability "
            "or guarantee room allocation. Please contact "
            "PNEC administration for current information."
        )

    # Find candidate records
    candidate_indices = data[
        data["category"].isin(categories)
    ].index.tolist()

    # For general admission questions, exclude hostel application records
    if "Admissions" in categories and "hostel" not in cleaned_input:

        candidate_indices = [
            i for i in candidate_indices
            if "hostel" not in cleaned_questions[i]
        ]

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

    print("Predicted Category:", categories)
    print("Matched Question:", data.iloc[best_index]["question"])
    print("Similarity Score:", round(best_score, 3))
    print("Matched Category:", data.iloc[best_index]["category"])

    # Similarity threshold
    if best_score >= 0.35:
        return data.iloc[best_index]["answer"]

    return "Sorry, I could not find an answer to your question."

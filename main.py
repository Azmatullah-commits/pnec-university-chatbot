from chatbot import find_answer

print("Welcome to PNEC University Chatbot!")

while True:

    question = input("\nAsk your question (or type exit): ")

    if question.lower() == "exit":
        print("Thank you for using the chatbot!")
        break

    answer = find_answer(question)

    print("Chatbot:", answer)
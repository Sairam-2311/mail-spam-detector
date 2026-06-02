import pickle

model = pickle.load(open("spam_model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    email = input("\nEnter email text (or type quit): ")

    if email.lower() == "quit":
        break
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)
    print("Prediction:", prediction[0])
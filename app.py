from flask import Flask, render_template, request
import pickle
import os
 
app = Flask(__name__)
try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    model = None
    vectorizer = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None
 
@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html",
            error="Model not loaded. Check server logs.")
 
    try:
        email = request.form.get("email", "").strip()
 
        if not email:
            return render_template("index.html",
                error="Please enter some email text.")
 
        if len(email) > 10000:
            return render_template("index.html",
                error="Input too long. Max 10,000 characters.")
 
        email_vector = vectorizer.transform([email])
        prediction = model.predict(email_vector)
 
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(email_vector)
            confidence = round(max(proba[0]) * 100, 2)
 
        return render_template(
            "index.html",
            prediction=prediction[0],
            confidence=confidence,
            email=email
        )
 
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("index.html",
            error="Something went wrong. Please try again.")
 
if __name__ == "__main__":
    app.run(debug=True)
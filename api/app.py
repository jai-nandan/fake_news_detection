from flask import Flask, request, render_template
import pickle
import os

# Base directory (so paths work on Vercel)
BASE_DIR = os.path.dirname(__file__)

# Load vectorizer and model from project root
with open(os.path.join(BASE_DIR, "../vectorizer.pkl"), "rb") as f:
    vector = pickle.load(f)

with open(os.path.join(BASE_DIR, "../finalized_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Flask app (templates + static outside /api/)
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        news = str(request.form["news"])
        prediction = model.predict(vector.transform([news]))[0]
        return render_template(
            "prediction.html",
            prediction_text=f"News headline is -> {prediction}"
        )
    return render_template("prediction.html")

# Required for Vercel
def handler(request, response):
    return app(request, response)

if __name__ == "__main__":
    app.run(debug=True)

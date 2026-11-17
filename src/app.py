import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "src"))

from flask import Flask, request, jsonify, render_template
from predict import predict_ticket
from responses import get_suggested_reply

app = Flask(
    __name__,
    template_folder=str(_project_root / "src" / "templates"),
    static_folder=str(_project_root / "src" / "static"),
)

def _get_model_name():
    try:
        import joblib
        pipeline = joblib.load(_project_root / "model" / "ticket_classifier.joblib")
        clf = pipeline.steps[-1][1]
        clf_name = type(clf).__name__
        name_map = {
            "MultinomialNB": "Multinomial Naive Bayes",
            "LogisticRegression": "Logistic Regression",
        }
        return f"TF-IDF with {name_map.get(clf_name, clf_name)}"
    except Exception:
        return "TF-IDF text classifier"


_model_name = _get_model_name()


@app.route("/")
def index():
    return render_template("index.html", model_name=_model_name)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"]

    if not isinstance(text, str):
        return jsonify({"error": "The 'text' field must be a string."}), 400

    stripped = text.strip()
    if stripped == "":
        return jsonify({"error": "Please enter a ticket description."}), 400

    if len(stripped) > 1000:
        return jsonify({"error": "Ticket description must be 1000 characters or fewer."}), 400

    try:
        category, confidence, top_3 = predict_ticket(stripped)
        low_confidence = confidence < 50
        suggested_reply = get_suggested_reply(category)

        top3_list = [{"category": cat, "probability": round(prob, 1)} for cat, prob in top_3]

        return jsonify({
            "category": category,
            "confidence": round(confidence, 1),
            "top3": top3_list,
            "low_confidence": low_confidence,
            "suggested_reply": suggested_reply,
        })
    except Exception as exc:
        print(f"Prediction error: {exc}", flush=True)
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    print("Starting Ticket Classifier server at http://127.0.0.1:5000", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
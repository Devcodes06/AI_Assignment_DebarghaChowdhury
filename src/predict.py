from pathlib import Path
import sys
import joblib
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from preprocessing import clean_text

_model = None
_labels = None


def _load_model():
    global _model, _labels
    if _model is None:
        model_path = project_root / "model" / "ticket_classifier.joblib"
        labels_path = project_root / "model" / "labels.txt"
        _model = joblib.load(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _model, _labels


def predict_ticket(text: str):
    model, labels = _load_model()

    cleaned = clean_text(text)

    if cleaned == "":
        return "Unknown", 0.0, []

    proba = model.predict_proba([cleaned])[0]
    predicted_idx = int(np.argmax(proba))
    predicted_category = labels[predicted_idx]
    confidence = round(float(proba[predicted_idx]) * 100, 1)

    top3_indices = np.argsort(proba)[-3:][::-1]
    top_3 = [(labels[i], round(float(proba[i]) * 100, 1)) for i in top3_indices]

    return predicted_category, confidence, top_3


def run_cli():
    print("=" * 50)
    print("  Ticket Category Predictor")
    print("=" * 50)
    print("Enter ticket descriptions to predict their category.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Enter Ticket Description (or type quit to exit): ").strip()
        except EOFError:
            break

        if text.lower() == "quit":
            print("Goodbye!")
            break

        if text == "":
            print("Please enter a description.\n")
            continue

        category, confidence, top_3 = predict_ticket(text)

        print(f"Predicted Category: {category}")
        print(f"Confidence: {confidence}%")
        if confidence < 50:
            print("Flag: Low confidence - please review manually")
        print("Top 3 probabilities:")
        for cat, prob in top_3:
            print(f"  {cat}: {prob}%")
        print()


if __name__ == "__main__":
    run_cli()
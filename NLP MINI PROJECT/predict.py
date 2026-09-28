"""
predict.py
----------
Inference utility for:
"Crowd Incident Text Categorization Using Natural Language Processing"

Usage:
  1) Command-line argument:
     python predict.py "Massive fire erupting from generator room near stage"

  2) Interactive prompt:
     python predict.py
"""

import sys
import joblib
import numpy as np
from preprocessing import preprocess_text

def load_system(model_file="models/logistic_regression.pkl", vectorizer_file="models/tfidf_vectorizer.pkl"):
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        print("[ERROR] Trained models not found in 'models/' directory.")
        print("Please run: python train_and_evaluate.py first.")
        sys.exit(1)


def predict_incident(text: str, model, vectorizer):
    """Predicts the category of an incident and displays confidence breakdown."""
    if not text.strip():
        print("[WARN] Empty input received.")
        return

    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    classes = model.classes_
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    print("\n" + "=" * 65)
    print(f"INPUT INCIDENT : \"{text}\"")
    print(f"PROCESSED NLP  : \"{processed}\"")
    print(f"PREDICTED CLASS: >>> {prediction} <<<")
    if probabilities is not None:
        confidence = np.max(probabilities) * 100
        print(f"CONFIDENCE     : {confidence:.2f}%")
        print("-" * 65)
        print("CATEGORY PROBABILITY DISTRIBUTION:")

        # Sort classes by probability descending
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            cat = classes[idx]
            prob = probabilities[idx] * 100
            bar = "#" * int(prob // 4)
            print(f"  {cat:<35}: {prob:5.2f}% | {bar}")
    print("=" * 65 + "\n")


def main():
    model, vectorizer = load_system()

    if len(sys.argv) > 1:
        # Text provided via command line
        input_text = " ".join(sys.argv[1:])
        predict_incident(input_text, model, vectorizer)
    else:
        # Interactive mode
        print("=" * 65)
        print("  CROWD INCIDENT TEXT CATEGORIZATION - LIVE INFERENCE")
        print("  Type any incident alert or description (or 'exit' to quit)")
        print("=" * 65)
        while True:
            try:
                user_input = input("\nEnter incident report: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting classifier. Stay safe!")
                    break
                if not user_input:
                    continue
                predict_incident(user_input, model, vectorizer)
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()

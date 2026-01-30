import torch
import joblib
import numpy as np
import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)


# TF-IDF + Logistic Regression Detector

class TfidfLRDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def score(self, prompt: str) -> float:
        return self.model.predict_proba([prompt])[0][1]



# DistilBERT Detector

class DistilBERTDetector:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def score(self, prompt: str) -> float:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        return probs[0][1].item()



# Decision Logic

def classify(score: float, threshold: float) -> str:
    return "MALICIOUS" if score >= threshold else "BENIGN"


def ensemble_decision(tfidf_score, bert_score,
                      tfidf_thresh=0.7,
                      bert_thresh=0.6):
    return "MALICIOUS" if (
        tfidf_score >= tfidf_thresh or
        bert_score >= bert_thresh
    ) else "BENIGN"



# Interactive Prompt Analyzer

def analyze_prompt(prompt, tfidf_detector, bert_detector):
    tfidf_score = tfidf_detector.score(prompt)
    bert_score = bert_detector.score(prompt)

    tfidf_decision = classify(tfidf_score, 0.7)
    bert_decision = classify(bert_score, 0.6)
    ensemble = ensemble_decision(tfidf_score, bert_score)

    print("\n" + "=" * 70)
    print("PROMPT:")
    print(prompt)
    print("-" * 70)
    print(f"TF-IDF + LR score : {tfidf_score:.4f} → {tfidf_decision}")
    print(f"DistilBERT score : {bert_score:.4f} → {bert_decision}")
    print("-" * 70)
    print(f"FINAL (ENSEMBLE) : {ensemble}")
    print("=" * 70)



# Main Loop

if __name__ == "__main__":

    print("\n Prompt Injection Detection System")
    print("Type a prompt to analyze, or 'exit' to quit.\n")

    tfidf_detector = TfidfLRDetector(
        "prompt_injection_classifier.joblib"
    )

    bert_detector = DistilBERTDetector(
        "distilbert_prompt_injection_detector"
    )

    while True:
        user_prompt = input(">>> ")

        if user_prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        analyze_prompt(
            user_prompt,
            tfidf_detector,
            bert_detector
        )

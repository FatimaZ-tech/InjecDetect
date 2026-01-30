# InjecDetect: An ML-Based Prompt Injection Detection System

**InjecDetect** is a machine learning based interactive prompt injection detection system designed to identify adversarial prompts submitted to Large Language Models (LLMs). Prompt injection and jailbreak attacks attempt to override system instructions through carefully crafted user inputs.

The system uses a dual machine learning approach, combining both classical and modern NLP models to analyze user provided prompts in real time and assess their likelihood of being malicious.

---

## What InjecDetect Does

1. Accepts a user provided prompt for analysis
2. Scores the prompt using a lexical ML model (TF-IDF + Logistic Regression)
3. Scores the prompt using a semantic AI model (DistilBERT) 
4. Combines and displays detection results in real time

---

## Key Objectives

- Detect prompt injection and jailbreak attempts in user inputs
- Provide real-time risk scoring for user provided prompts  
- Compare lexical and semantic detection approaches  
- Demonstrate a layered AI security design using multiple complementary models

---

## Models & Techniques Used

The system uses two complementary ML models, each capturing different characteristics of malicious prompts:

### TF-IDF + Logistic Regression (Lexical Detector)

- Extracts word and n-gram level features using **TF-IDF**
- Trained with **Logistic Regression** for binary classification
- Strong at detecting explicit jailbreak phrases and known attack templates
- Fast and interpretable as a reliable baseline

### DistilBERT (Semantic Detector)

- Transformer based language model fine-tuned for binary classification
- Captures contextual and semantic intent
- Better at detecting paraphrased or indirect prompt injection attacks

---

## Dataset

The models were trained and evaluated using the **Malicious Prompt Detection Dataset (MPDD)**.

- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd/data

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/FatimaZ-tech/InjecDetect.git
cd InjecDetect
pip install -r requirements.txt
```

---

## Running InjecDetect

> Note: Trained model files must be present locally. See **Model Files & Setup**.

```bash
python injecdetect.py
```

Type any prompt at the prompt. Type `exit` or `quit` to stop.

---

## Model Files & Setup

Due to size limitations, trained model files are not included in this repository.

This project will not run unless the required models are available locally.  
Users must either:

- Train the models themselves or  
- Request the pre-trained model artifacts from the author.

This follows standard practices for managing large machine learning assets in public repositories.

---

## Limitations

- This system does not generate attacks; it detects them
- Detection performance depends on the diversity of training data
- Some subtle or novel attacks may still evade both models

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software for research, educational, and operational purposes, provided that the original copyright notice and license are included.

See the `LICENSE` file for full license text.

---

## Author

Developed by **Fatima Zakir**.

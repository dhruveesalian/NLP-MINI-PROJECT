# Crowd Incident Text Categorization Using Natural Language Processing

An end-to-end Natural Language Processing (NLP) system designed to detect, classify, and triage crowd safety incident descriptions into actionable public safety categories in real time.

---

## 📌 Project Overview
Large-scale public gatherings (festivals, stadiums, transit hubs, religious events) require real-time situational awareness. Eyewitness reports, emergency calls, social media messages, and marshal logs often contain unstructured text. This project applies NLP and Machine Learning to classify incident text into 6 critical operational categories:

1. **Stampede / Crushing Hazard**: Surges, bottlenecks, falling, suffocating, trampling.
2. **Violent Disorder / Brawl**: Physical fights, weapons, missile throwing, riots, hooliganism.
3. **Medical Emergency / Overheating**: Heatstroke, dehydration, cardiac arrest, fainting, asthma.
4. **Fire / Hazardous Smoke**: Pyrotechnics, electrical fires, toxic fumes, explosions, kitchen grease fires.
5. **Structural Failure / Collapse**: Barricade snap, grandstand/balcony collapse, scaffold failures.
6. **Normal Congestion / Safe Movement**: Orderly ingress/egress, safe queuing, routine movement.

---

## 📁 Repository Structure
```
NLP MINI PROJECT/
├── crowd_incidents.csv             # 360 balanced incident records across 6 categories
├── create_dataset.py               # Script to generate benchmark dataset
├── preprocessing.py                # NLP text cleaning, tokenization & lemmatization pipeline
├── train_and_evaluate.py           # Model training, evaluation metrics & visualization generator
├── predict.py                      # Interactive & CLI inference tool for new incident text
├── models/                         # Serialized models and vectorizer
│   ├── tfidf_vectorizer.pkl
│   ├── multinomial_naive_bayes.pkl
│   ├── logistic_regression.pkl
│   └── support_vector_machine_svm.pkl
├── outputs/                        # Evaluation charts, metrics & datasets
│   ├── model_comparison.csv
│   ├── model_performance_comparison.png
│   ├── confusion_matrices_all.png
│   ├── confusion_matrix_multinomial_naive_bayes.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_support_vector_machine_svm.png
│   ├── preprocessed_dataset.csv
│   └── sample_predictions.csv
└── README.md
```

---

## ⚙️ Installation & Prerequisites
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib
```

---

## 🚀 How to Run

### 1. Generate Dataset
```bash
python create_dataset.py
```

### 2. Train Models & Generate Visualizations
```bash
python train_and_evaluate.py
```
This script will:
- Preprocess incident texts using NLTK lemmatization and stopword filtering.
- Extract TF-IDF n-gram features (unigrams + bigrams).
- Train Multinomial Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).
- Evaluate Accuracy, Macro/Weighted Precision, Recall, and F1-Scores.
- Plot and save confusion matrices and comparison charts in `outputs/`.
- Export trained models to `models/`.

### 3. Classify New Incident Reports (CLI / Interactive)
Single text input:
```bash
python predict.py "Sudden crowd crush at gate 4, people being stepped on against the barrier!"
```
Interactive terminal mode:
```bash
python predict.py
```

---

## 📊 Experimental Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| **Logistic Regression** | **93.06%** | **0.9312** | **0.9306** | **0.9293** |
| **Support Vector Machine (SVM)** | **93.06%** | **0.9312** | **0.9306** | **0.9293** |
| **Multinomial Naive Bayes** | **91.67%** | **0.9199** | **0.9167** | **0.9172** |

All figures and tables are saved in `outputs/` for inclusion in the mini-project report.

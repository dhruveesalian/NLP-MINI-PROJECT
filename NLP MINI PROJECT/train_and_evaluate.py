"""
train_and_evaluate.py
---------------------
Core training, evaluation, and visualization pipeline for:
"Crowd Incident Text Categorization Using Natural Language Processing"

Workflow:
1. Loads crowd_incidents.csv
2. Preprocesses text with preprocessing.py
3. Splits into train/test sets (80/20 stratified)
4. Extracts TF-IDF features (unigrams + bigrams)
5. Trains Naive Bayes, Logistic Regression, and SVM
6. Evaluates Accuracy, Precision, Recall, F1-Score
7. Generates Confusion Matrices and Comparison Visualizations in outputs/
8. Exports trained models to models/ directory
9. Demonstrates live prediction on unseen incident descriptions
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress minor future scikit-learn probability warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_text

# Create output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Set visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def load_and_prepare_data(filepath="crowd_incidents.csv"):
    print("=" * 70)
    print("1. LOADING & PREPROCESSING DATASET")
    print("=" * 70)
    df = pd.read_csv(filepath)
    print(f"Total raw records loaded: {len(df)}")
    print(f"Categories ({df['category'].nunique()}):")
    for cat, count in df['category'].value_counts().items():
        print(f"  - {cat:<36}: {count} samples")

    print("\nApplying NLP text preprocessing (cleaning, tokenizing, lemmatizing)...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Save preprocessed dataset for verification
    df.to_csv("outputs/preprocessed_dataset.csv", index=False)
    print("Saved preprocessed dataset to 'outputs/preprocessed_dataset.csv'.\n")
    return df


def plot_confusion_matrix(cm, classes, model_name, filename):
    """Plots and saves an individual confusion matrix heatmap."""
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, weight='bold', pad=15)
    plt.ylabel('Actual Category', fontsize=11, weight='bold')
    plt.xlabel('Predicted Category', fontsize=11, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_combined_confusion_matrices(cms, classes, model_names, filename):
    """Plots all confusion matrices side-by-side for direct visual comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    cmaps = ['Purples', 'Greens', 'Blues']

    for idx, (cm, name, cmap) in enumerate(zip(cms, model_names, cmaps)):
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=classes,
            yticklabels=classes if idx == 0 else False,
            cbar=False,
            ax=axes[idx],
            linewidths=0.5
        )
        axes[idx].set_title(f"{name}", fontsize=13, weight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Category', fontsize=10, weight='bold')
        if idx == 0:
            axes[idx].set_ylabel('Actual Category', fontsize=10, weight='bold')
        axes[idx].tick_params(axis='x', rotation=40, labelsize=8)
        axes[idx].tick_params(axis='y', rotation=0, labelsize=8)

    plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(comparison_df, filename):
    """Plots grouped bar chart comparing performance metrics across models."""
    metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    plot_df = comparison_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("mako", len(metrics))
    ax = sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', palette=palette)

    plt.title("Crowd Incident Categorization - Model Performance Comparison", fontsize=14, weight='bold', pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, weight='bold')
    plt.xlabel("Machine Learning Model", fontsize=11, weight='bold')
    plt.ylim(0.70, 1.02)
    plt.legend(loc='lower right', frameon=True)

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8.5, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_pipeline():
    # 1. Load data
    df = load_and_prepare_data("crowd_incidents.csv")

    # 2. Train-Test Split (80% train, 20% test stratified)
    X = df['processed_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("=" * 70)
    print("2. DATASET SPLIT")
    print("=" * 70)
    print(f"Training set: {len(X_train)} samples (80%)")
    print(f"Testing set:  {len(X_test)} samples (20%)")

    # 3. TF-IDF Vectorization
    print("\n" + "=" * 70)
    print("3. TF-IDF VECTORIZATION")
    print("=" * 70)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF Feature Matrix: {X_train_tfidf.shape[1]} n-gram features extracted.")

    # Save vectorizer
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # 4. Initialize Models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    }

    results = []
    cms = []
    classes = sorted(list(y.unique()))

    print("\n" + "=" * 70)
    print("4. MODEL TRAINING & DETAILED EVALUATION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n---> Training: {name}")
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store in results
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec_macro, 4),
            "Recall (Macro)": round(rec_macro, 4),
            "F1-Score (Macro)": round(f1_macro, 4),
            "Precision (Weighted)": round(prec_weighted, 4),
            "Recall (Weighted)": round(rec_weighted, 4),
            "F1-Score (Weighted)": round(f1_weighted, 4),
        })

        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=classes, digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cms.append(cm)

        # Save individual confusion matrix
        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(cm, classes, name, f"outputs/confusion_matrix_{safe_filename}.png")

        # Save model
        joblib.dump(model, f"models/{safe_filename}.pkl")

    # 5. Visualizations & Summary
    print("=" * 70)
    print("5. PERFORMANCE COMPARISON & VISUALIZATION GENERATION")
    print("=" * 70)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv("outputs/model_comparison.csv", index=False)

    print("\nSummary Table:")
    print(comparison_df.to_string(index=False))

    # Plot combined charts
    plot_combined_confusion_matrices(cms, classes, list(models.keys()), "outputs/confusion_matrices_all.png")
    plot_performance_comparison(comparison_df, "outputs/model_performance_comparison.png")
    print("\nVisualizations saved to 'outputs/':")
    print("  - outputs/confusion_matrix_*.png (Individual heatmaps)")
    print("  - outputs/confusion_matrices_all.png (Combined grid)")
    print("  - outputs/model_performance_comparison.png (Grouped bar chart)")
    print("  - outputs/model_comparison.csv (Tabulated metrics)")

    # 6. Sample Live Predictions for New Incident Text
    print("\n" + "=" * 70)
    print("6. PREDICTIONS ON NEW UNSEEN INCIDENT ALERTS")
    print("=" * 70)

    unseen_incidents = [
        "Uncontrolled human surge at south tunnel exit, several fans fell and are being trampled against turnstiles!",
        "Violent clash near food court, two factions throwing broken glass bottles and metal chairs at each other.",
        "Paramedics urgently needed at gate 9, spectator collapsed with severe heat exhaustion and no pulse.",
        "Thick black smoke and flames coming from faulty food stall generator, crowd screaming and running away.",
        "Front metal barricade snapped in half and collapsed into pit, people fallen into ditch under pressure.",
        "Spectators calmly queuing at ticket inspection counter, moving through gates smoothly without delay."
    ]

    best_model_name = comparison_df.sort_values(by="F1-Score (Macro)", ascending=False).iloc[0]["Model"]
    best_model_file = best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    best_model = joblib.load(f"models/{best_model_file}.pkl")
    print(f"Using top-performing model: {best_model_name}\n")

    prediction_records = []
    for inc_text in unseen_incidents:
        proc = preprocess_text(inc_text)
        vec = vectorizer.transform([proc])
        pred = best_model.predict(vec)[0]
        prob = np.max(best_model.predict_proba(vec)) * 100

        print(f"Incident Text: \"{inc_text}\"")
        print(f"  -> Predicted Category: {pred}")
        print(f"  -> Confidence Score:   {prob:.2f}%\n")

        prediction_records.append({
            "Input Text": inc_text,
            "Predicted Category": pred,
            "Confidence (%)": round(prob, 2)
        })

    pd.DataFrame(prediction_records).to_csv("outputs/sample_predictions.csv", index=False)
    print("Saved sample predictions to 'outputs/sample_predictions.csv'.")
    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()

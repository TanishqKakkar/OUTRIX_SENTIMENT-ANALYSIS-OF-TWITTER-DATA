# model_comparison.py

import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Paths
# =========================
lstm_model_path = r"D:\major projectt\best_lstm_model.h5"
tfidf_vectorizer_path = r"D:\major projectt\tfidf_vectorizer.pkl"
label_encoder_path = r"D:\major projectt\label_encoder.pkl"
bert_model_path = r"D:\major projectt\bert_sentiment_model"
validation_csv_path = r"D:\major projectt\twitter_validation.csv"

# =========================
# Load validation dataset
# =========================
df = pd.read_csv(validation_csv_path, header=None)
df.columns = ['id', 'platform', 'label', 'text']  # Assign proper column names
texts = df['text'].tolist()
true_labels = df['label'].tolist()

# =========================
# Load Label Encoder
# =========================
with open(label_encoder_path, "rb") as f:
    le = pickle.load(f)
class_names = le.classes_

# =========================
# Load LSTM + TF-IDF model
# =========================
print("Loading LSTM + TF-IDF model...")
lstm_model = load_model(lstm_model_path)
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

def predict_lstm(texts):
    X = tfidf_vectorizer.transform(texts).toarray()
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # reshape for LSTM
    preds = lstm_model.predict(X)
    return le.inverse_transform(preds.argmax(axis=1))

# =========================
# Load fine-tuned BERT model
# =========================
print("Loading BERT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

def predict_bert(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    return le.inverse_transform(preds)

# =========================
# Predictions
# =========================
print("Predicting with LSTM + TF-IDF...")
lstm_preds = predict_lstm(texts)

print("Predicting with BERT...")
bert_preds = predict_bert(texts)

# =========================
# Metrics
# =========================
lstm_acc = accuracy_score(true_labels, lstm_preds)
bert_acc = accuracy_score(true_labels, bert_preds)
lstm_f1 = f1_score(true_labels, lstm_preds, average='weighted')
bert_f1 = f1_score(true_labels, bert_preds, average='weighted')

# =========================
# Print comparison table
# =========================
print("\n==============================")
print("Model Comparison on Twitter Validation Set")
print("==============================")
print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
print(f"{'-'*40}")
print(f"{'LSTM + TF-IDF':<20} {lstm_acc:.4f}    {lstm_f1:.4f}")
print(f"{'BERT':<20} {bert_acc:.4f}    {bert_f1:.4f}")
print("==============================\n")

# =========================
# Show top 10 disagreements
# =========================
disagree_idx = [i for i, (l, b) in enumerate(zip(lstm_preds, bert_preds)) if l != b]
print("Top 10 tweets where LSTM and BERT disagree:")
for idx in disagree_idx[:10]:
    print(f"\nTweet: {texts[idx]}")
    print(f"LSTM Prediction: {lstm_preds[idx]} | BERT Prediction: {bert_preds[idx]} | True Label: {true_labels[idx]}")

# =========================
# 1. Accuracy & F1 comparison bar chart
# =========================
plt.figure(figsize=(6,4))
metrics = ['Accuracy', 'F1-Score']
lstm_scores = [lstm_acc, lstm_f1]
bert_scores = [bert_acc, bert_f1]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, lstm_scores, width, label='LSTM + TF-IDF')
plt.bar(x + width/2, bert_scores, width, label='BERT')
plt.ylim(0,1)
plt.ylabel('Score')
plt.title('Model Comparison on Validation Set')
plt.xticks(x, metrics)
plt.legend()
plt.show()

# =========================
# 2. Confusion matrices
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

plot_confusion_matrix(true_labels, lstm_preds, 'LSTM + TF-IDF')
plot_confusion_matrix(true_labels, bert_preds, 'BERT')

# =========================
# 3. Disagreement distribution
# =========================
plt.figure(figsize=(6,4))
disagree_counts = [np.sum(lstm_preds != true_labels), np.sum(bert_preds != true_labels)]
plt.bar(['LSTM + TF-IDF', 'BERT'], disagree_counts, color=['orange','green'])
plt.ylabel('Number of Misclassifications')
plt.title('Misclassifications vs True Labels')
plt.show()

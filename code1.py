import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import re
import emoji

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification

train_df = pd.read_csv(r"D:\major prroject puru\twitter_training.csv", header=None)
val_df = pd.read_csv(r"D:\major prroject puru\twitter_validation.csv", header=None)

print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
train_df.head()
val_df.head()
train_df.columns = ['tweet_id', 'entity', 'sentiment', 'content']
val_df.columns = ['tweet_id', 'entity', 'sentiment', 'content']
train_df.columns
train_df.duplicated().sum()
train_df = train_df.drop_duplicates().reset_index(drop=True)
train_df.isna().sum()
train_df[train_df['content'].isna()]
train_df = train_df.dropna().reset_index(drop=True)
def preprocess_text(text):
    text = str(text).lower()                                # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)     # remove urls
    text = re.sub(r"@\w+|#\w+", '', text)                   # remove mentions & hashtags
    text = re.sub(r"[0-9]+", '', text)                      # remove numbers
    text = re.sub(r"[^\w\s]", '', text)                     # remove punctuation
    text = emoji.replace_emoji(text, replace='')            # remove emojis
    text = re.sub(r"\s+", ' ', text).strip()                # remove extra spaces
    return text
train_df['content'] = train_df['content'].apply(preprocess_text)
val_df['content']   = val_df['content'].apply(preprocess_text)
train_df.head()
label_encoder = LabelEncoder()
train_df['sentiment'] = label_encoder.fit_transform(train_df['sentiment'])
val_df['sentiment'] = label_encoder.transform(val_df['sentiment'])
# Save Label Encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
tfidf = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf.fit_transform(train_df['content']).toarray()
x_val_tfidf = tfidf.transform(val_df['content']).toarray()
# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
    y_train = train_df['sentiment'].values
y_val = val_df['sentiment'].values
x_train_lstm = np.expand_dims(x_train_tfidf, axis=1)
x_val_lstm = np.expand_dims(x_val_tfidf, axis=1)
num_classes = len(label_encoder.classes_)

lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2])))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(num_classes, activation='softmax'))
lstm_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2, 
    min_lr=1e-5
)
history = lstm_model.fit(
    x_train_lstm, y_train,
    validation_data=(x_val_lstm, y_val),
    epochs=100,    
    batch_size=64,
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
best_lstm_model = load_model('best_lstm_model.h5')
y_pred = best_lstm_model.predict(x_val_lstm)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))
conf_matrix = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
texts = [
    "I love this product!",    # positive
    "This is terrible."       # negative
]
x_input = tfidf.transform(texts).toarray()
x_input = np.expand_dims(x_input, axis=1)
y_pred_probs = best_lstm_model.predict(x_input)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
predicted_labels = label_encoder.inverse_transform(y_pred_classes)

print(predicted_labels)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization
x_train_bert = tokenizer(list(train_df['content']), truncation=True, padding=True, max_length=128)
x_val_bert = tokenizer(list(val_df['content']), truncation=True, padding=True, max_length=128)
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_ds = SentimentDataset(x_train_bert, y_train)
val_ds = SentimentDataset(x_val_bert, y_val)
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=10, 
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
results = trainer.evaluate()
print(results)
trainer.save_model("./bert_model")
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ----------------------------
# 1. Load Dataset (No Header)
# ----------------------------
train_path = r"D:\major projectt\twitter_training.csv"
val_path = r"D:\major projectt\twitter_validation.csv"

train_df = pd.read_csv(train_path, header=None)
val_df = pd.read_csv(val_path, header=None)

# Assign column names
train_df.columns = ["id", "entity", "sentiment", "text"]
val_df.columns = ["id", "entity", "sentiment", "text"]

# Drop rows with missing text
train_df = train_df.dropna(subset=["text"])
val_df = val_df.dropna(subset=["text"])

# ----------------------------
# 2. Encode Sentiment Labels
# ----------------------------
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["sentiment"])
val_df["label"] = le.transform(val_df["sentiment"])

print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ----------------------------
# 3. Prepare BERT Tokenizer
# ----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 128

class TwitterDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        enc = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = TwitterDataset(train_df)
val_dataset = TwitterDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ----------------------------
# 4. Load BERT Model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(train_df["label"].unique())
).to(device)

# ----------------------------
# 5. Optimizer & Scheduler
# ----------------------------
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

# ----------------------------
# 6. Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\n---- Epoch {epoch+1}/{EPOCHS} ----")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Training Loss: {total_loss/len(train_loader):.4f}")

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {correct/total:.4f}")
# ----------------------------
# 7. Save Model
# ----------------------------
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")

print("\nModel saved in 'bert_sentiment_model/'")
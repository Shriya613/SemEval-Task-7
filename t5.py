import warnings
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LEN = 512

# Load the CSV files
print("Loading CSV files...")
posts_df = pd.read_csv('/home/klarocca/nlp243/semeval/posts.csv')
fact_checks_df = pd.read_csv('/home/klarocca/nlp243/semeval/fact_checks.csv')
pairs_df = pd.read_csv('/home/klarocca/nlp243/semeval/pairs.csv')

# Create positive examples
print("Creating positive examples...")
data = pd.merge(
    pd.merge(pairs_df, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
data['label'] = 1

# Generate negative examples
print("Creating negative examples...")
negative_pairs = pairs_df.copy()
negative_pairs['fact_check_id'] = negative_pairs['fact_check_id'].sample(frac=1, random_state=42).reset_index(drop=True)
negative_data = pd.merge(
    pd.merge(negative_pairs, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
negative_data['label'] = 0

# Combine datasets
print("Combining datasets...")
data = pd.concat([data, negative_data]).reset_index(drop=True)

# Normalize text
print("Normalizing text...")
data['ocr'] = data['ocr'].str.lower().str.strip()
data['claim'] = data['claim'].str.lower().str.strip()

# Split data into train, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

# Dataset class
class T5Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.inputs = []
        self.attention_masks = []
        self.labels = []

        for _, row in dataframe.iterrows():
            text = f"ocr: {row['ocr']} claim: {row['claim']}"
            label = "relevant" if row['label'] == 1 else "not relevant"

            inputs = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = tokenizer(
                label,
                max_length=2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(inputs.input_ids.squeeze())
            self.attention_masks.append(inputs.attention_mask.squeeze())
            self.labels.append(labels.input_ids.squeeze())

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input_ids": self.inputs[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index],
        }

# Initialize tokenizer and model
print("Initializing tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Create DataLoaders
print("Creating DataLoaders...")
train_dataset = T5Dataset(train_data, tokenizer, max_len=MAX_LEN)
val_dataset = T5Dataset(val_data, tokenizer, max_len=MAX_LEN)
test_dataset = T5Dataset(test_data, tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch + 1}/{EPOCHS}...")
    model.train()
    train_loss = 0

    for i, batch in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Processing batch {i}/{len(train_loader)}...")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {train_loss / len(train_loader):.4f}")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Validation Loss: {val_loss / len(val_loader):.4f}")

# Test the model
print("Testing model on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        labels = [tokenizer.decode(label, skip_special_tokens=True) for label in batch["labels"]]

        all_preds.extend(preds)
        all_labels.extend(labels)

# Evaluate results
binary_preds = [1 if pred == "relevant" else 0 for pred in all_preds]
binary_labels = [1 if label == "relevant" else 0 for label in all_labels]

print("Classification Report:")
print(classification_report(binary_labels, binary_preds))
print(f"Test Accuracy: {accuracy_score(binary_labels, binary_preds):.4f}")

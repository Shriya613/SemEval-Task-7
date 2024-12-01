import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Text Cleaning Function
def clean_text(text):
    text = text.str.replace(r"<.*?>", "", regex=True)  # Remove HTML tags
    text = text.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)  # Remove non-alphanumeric characters
    text = text.str.lower().str.strip()  # Convert to lowercase and strip leading/trailing spaces
    return text

# Dataset Class
class CSVClaimDatasetTriplet(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(
            row['ocr'], row['claim'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.float),
            'post_id': row['post_id'],
            'fact_check_id': row['fact_check_id']
        }

# Model Creation
class SimilarityModel(nn.Module):
    def __init__(self, base_model_name="xlm-roberta-base"):
        super(SimilarityModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)  # Add a linear layer for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Pool the hidden states
        logits = self.classifier(pooled_output)  # Pass through the classifier
        return logits


# Training Function
def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)  # Ensure labels have shape (batch_size, 1)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)  # Directly get logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation Function with MRR and Precision@K
def validate_model(model, dataloader, loss_fn, device, k):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []  # Collect all scores for Precision@K and MRR
    all_post_ids = []  # Collect post_ids for MRR
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)  # Ensure labels have shape (batch_size, 1)

            logits = model(input_ids, attention_mask)  # Directly get logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Collect scores and post_ids for MRR and Precision@K
            all_scores.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_post_ids.extend(batch['post_id'].tolist())

    accuracy = accuracy_score(all_labels, all_preds)

    # Compute MRR (Mean Reciprocal Rank)
    mrr = compute_mrr(all_scores, all_labels, all_post_ids)

    # Compute Precision@K
    precision_at_k = compute_precision_at_k(all_scores, all_labels, k)

    return total_loss / len(dataloader), accuracy, mrr, precision_at_k

# MRR Calculation
def compute_mrr(scores, labels, post_ids):
    df = pd.DataFrame({
        'post_id': post_ids,
        'score': scores,
        'label': labels
    })
    mrr = 0
    grouped = df.groupby('post_id')
    for _, group in grouped:
        group = group.sort_values(by='score', ascending=False)  # Rank by score
        reciprocal_rank = 0
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            if row['label'] == 1:  # Found relevant item
                reciprocal_rank = 1 / rank
                break
        mrr += reciprocal_rank
    return mrr / len(grouped)

# Precision@K Calculation
def compute_precision_at_k(scores, labels, k):
    sorted_indices = np.argsort(scores)[::-1]  # Descending order of scores
    top_k_indices = sorted_indices[:k]
    relevant_at_k = np.sum(np.array(labels)[top_k_indices])
    return relevant_at_k / k

# Save Predictions to CSV
def save_predictions_to_csv(model, dataloader, output_path, device):
    model.eval()
    predictions = []
    post_ids = []
    fact_check_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            predicted_labels = (scores > 0.5).astype(int)

            post_ids.extend(batch['post_id'].tolist())
            fact_check_ids.extend(batch['fact_check_id'].tolist())
            predictions.extend(predicted_labels.tolist())

    results = pd.DataFrame({
        'post_id': post_ids,
        'fact_check_id': fact_check_ids,
        'predicted_label': predictions
    })

    relevant_fact_checks = (
        results[results['predicted_label'] == 1]
        .groupby('post_id')['fact_check_id']
        .apply(list)
        .reset_index()
    )
    relevant_fact_checks.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# Main Code
if __name__ == "__main__":
    # Load and preprocess data
    pairs_df = pd.read_csv('/home/ikulkar1/semeval/pairs.csv', encoding='utf-8', engine='python')
    fact_checks_df = pd.read_csv('/home/ikulkar1/semeval/fact_checks.csv', encoding='utf-8', engine='python')
    posts_df = pd.read_csv('/home/ikulkar1/semeval/posts.csv', encoding='utf-8', engine='python')

    merged_data = pd.merge(
        pd.merge(pairs_df, posts_df[['post_id', 'ocr']], on='post_id'),
        fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
    )
    merged_data['label'] = 1
    negative_data = merged_data.copy()
    negative_data['fact_check_id'] = negative_data['fact_check_id'].sample(frac=1).reset_index(drop=True)
    negative_data['label'] = 0
    data = pd.concat([merged_data, negative_data]).reset_index(drop=True)

    data['ocr'] = clean_text(data['ocr'])
    data['claim'] = clean_text(data['claim'])

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    dataset = CSVClaimDatasetTriplet(data, tokenizer)

    train_size = int(0.85 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SimilarityModel().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    num_epochs = 25
    k = 10  # Precision@K

    with open("v4_outputs.txt", "w") as f:
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_accuracy, val_mrr, val_precision_at_k = validate_model(model, val_loader, loss_fn, device, k=k)

            # Log metrics to file
            f.write(f"Epoch {epoch + 1}/{num_epochs}:\n")
            f.write(f"Train Loss: {train_loss}\n")
            f.write(f"Validation Loss: {val_loss}\n")
            f.write(f"Validation Accuracy: {val_accuracy}\n")
            f.write(f"Validation MRR: {val_mrr}\n")
            f.write(f"Validation Precision@{k}: {val_precision_at_k}\n")
            f.write("-" * 40 + "\n")

            # Print metrics to terminal
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")
            print(f"Validation Accuracy: {val_accuracy}")
            print(f"Validation MRR: {val_mrr}")
            print(f"Validation Precision@{k}: {val_precision_at_k}")

            if (epoch + 1) % 2 == 0:
                torch.save(model.state_dict(), f'./model_v4_epoch_{epoch+1}.pth')

    # Save final model predictions
    save_predictions_to_csv(model, test_loader, 'v4_predictions.csv', device)

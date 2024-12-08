import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Data Preparation
pairs_df = pd.read_csv('/Users/yshriyasravani/Documents/SemEval_Task7_mine/SemEval-Task-7_mine/pairs.csv')
fact_checks_df = pd.read_csv('/Users/yshriyasravani/Documents/SemEval_Task7_mine/SemEval-Task-7_mine/fact_checks.csv')
posts_df = pd.read_csv('/Users/yshriyasravani/Documents/SemEval_Task7_mine/SemEval-Task-7_mine/posts.csv')

# Positive Examples
positive_data = pd.merge(
    pd.merge(pairs_df, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
positive_data['label'] = 1

# Negative Examples
negative_pairs = pairs_df.copy()
negative_pairs['fact_check_id'] = negative_pairs['fact_check_id'].sample(frac=1, random_state=42).reset_index(drop=True)
negative_data = pd.merge(
    pd.merge(negative_pairs, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
negative_data['label'] = 0

# Combine and Preprocess
data = pd.concat([positive_data, negative_data]).reset_index(drop=True)
data['ocr'] = data['ocr'].str.lower().str.strip()
data['claim'] = data['claim'].str.lower().str.strip()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
combined_text = pd.concat([train_data['ocr'], train_data['claim']])
tfidf_vectorizer.fit(combined_text)

X_train_ocr = tfidf_vectorizer.transform(train_data['ocr']).toarray()
X_train_claim = tfidf_vectorizer.transform(train_data['claim']).toarray()
y_train = train_data['label'].values

X_test_ocr = tfidf_vectorizer.transform(test_data['ocr']).toarray()
X_test_claim = tfidf_vectorizer.transform(test_data['claim']).toarray()
y_test = test_data['label'].values

# Step 2: Dataset
class ClaimDataset(Dataset):
    def __init__(self, ocr, claim, labels):
        self.ocr = torch.tensor(ocr, dtype=torch.float32)
        self.claim = torch.tensor(claim, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ocr[idx], self.claim[idx], self.labels[idx]

train_dataset = ClaimDataset(X_train_ocr, X_train_claim, y_train)
test_dataset = ClaimDataset(X_test_ocr, X_test_claim, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 3: BiGRU with Multi-Head Attention
class BiGRUMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(BiGRUMultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_ocr = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru_claim = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2 * hidden_dim * 2, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, ocr, claim):
        ocr = ocr.unsqueeze(1)
        claim = claim.unsqueeze(1)
        
        gru_ocr_out, _ = self.gru_ocr(ocr)
        gru_claim_out, _ = self.gru_claim(claim)
        
        # Apply Multi-Head Attention
        attn_ocr_out, _ = self.multihead_attn(gru_ocr_out, gru_ocr_out, gru_ocr_out)
        attn_claim_out, _ = self.multihead_attn(gru_claim_out, gru_claim_out, gru_claim_out)
        
        # Combine outputs
        combined = torch.cat((attn_ocr_out.mean(dim=1), attn_claim_out.mean(dim=1)), dim=1)
        
        x = self.fc1(combined)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Model Initialization
input_dim = X_train_ocr.shape[1]
hidden_dim = 64
output_dim = 2
num_heads = 4

model = BiGRUMultiHeadAttention(input_dim, hidden_dim, output_dim, num_heads)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

# Step 4: Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for ocr, claim, labels in train_loader:
        ocr, claim, labels = ocr.to(device), claim.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(ocr, claim)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Step 5: Evaluation
model.eval()
all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for ocr, claim, labels in test_loader:
        ocr, claim, labels = ocr.to(device), claim.to(device), labels.to(device)
        outputs = model(ocr, claim)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds))
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# Step 6: Save Predictions
predictions_df = pd.DataFrame({
    'OCR': test_data['ocr'].values,
    'Claim': test_data['claim'].values,
    'True Label': all_labels,
    'Predicted Label': all_preds
})

predictions_df.to_csv('bigru_with_attention_predictions.csv', index=False)
print("Predictions saved to 'bigru_with_attention_predictions.csv'.")

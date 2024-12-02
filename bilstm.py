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
pairs_df = pd.read_csv('pairs.csv')
fact_checks_df = pd.read_csv('fact_checks.csv')
posts_df = pd.read_csv('posts.csv')

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

# Step 3: BiLSTM with Attention
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_ocr = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_claim = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Attention
        self.attention = nn.Linear(2 * hidden_dim, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2 * hidden_dim * 2, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def attention_weights(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context
    
    def forward(self, ocr, claim):
        ocr = ocr.unsqueeze(1)
        claim = claim.unsqueeze(1)
        
        lstm_ocr_out, _ = self.lstm_ocr(ocr)
        lstm_claim_out, _ = self.lstm_claim(claim)
        
        context_ocr = self.attention_weights(lstm_ocr_out)
        context_claim = self.attention_weights(lstm_claim_out)
        
        combined = torch.cat((context_ocr, context_claim), dim=1)
        x = self.fc1(combined)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Model
input_dim = X_train_ocr.shape[1]
hidden_dim = 64
output_dim = 2

model = BiLSTMAttentionModel(input_dim, hidden_dim, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Success@K Metric
def success_at_k(predictions, labels, k=3):
    correct = 0
    for i in range(len(predictions)):
        top_k = predictions[i].argsort(descending=True)[:k]
        if labels[i] in top_k:
            correct += 1
    return correct / len(predictions)

# Training Loop
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

# Evaluation and Metrics
model.eval()
all_preds, all_probs, all_labels, ocrs, claims = [], [], [], [], []

with torch.no_grad():
    for ocr, claim, labels in test_loader:
        ocr, claim, labels = ocr.to(device), claim.to(device), labels.to(device)
        outputs = model(ocr, claim)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        ocrs.extend(ocr.cpu().numpy())
        claims.extend(claim.cpu().numpy())

# Success@K Metric
k = 3
success_k = success_at_k(torch.tensor(all_probs), torch.tensor(all_labels), k=k)
print(f"Success@{k}: {success_k:.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds))
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# Save to CSV
predictions_df = pd.DataFrame({
    'OCR': [
        ' '.join(tfidf_vectorizer.inverse_transform(row.reshape(1, -1))[0])
        for row in X_test_ocr
    ],
    'Claim': [
        ' '.join(tfidf_vectorizer.inverse_transform(row.reshape(1, -1))[0])
        for row in X_test_claim
    ],
    'True Label': all_labels,
    'Predicted Label': all_preds,
    'Prediction Probability': all_probs
})

# Save the DataFrame to a CSV file
predictions_df.to_csv('bilstmk_predictions.csv', index=False)

print("Predictions saved to 'bilstmk_predictions.csv'.")

# Filter predictions where the predicted label is 1
fact_check_ids = test_data.iloc[:len(all_preds)]['fact_check_id'].values

predictions_df = pd.DataFrame({
    'Fact Check ID': fact_check_ids,
    'True Label': all_labels,
    'Predicted Label': all_preds,
    'Prediction Probability': all_probs
})

# Save only rows where the predicted label is 1
predictions_df = predictions_df[predictions_df['Predicted Label'] == 1]

# Save the DataFrame to a CSV file
predictions_df[['Fact Check ID']].to_csv('relevant_fact_checks.csv', index=False)

print("Relevant fact checks saved to 'relevant_fact_checks.csv'.")

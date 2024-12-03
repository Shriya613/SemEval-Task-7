from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd

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

# Generate better negative examples
print("Creating better negative examples...")
negative_pairs = pairs_df.copy()
negative_pairs['fact_check_id'] = negative_pairs['fact_check_id'].sample(frac=1, random_state=42).reset_index(drop=True)
negative_data = pd.merge(
    pd.merge(negative_pairs, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
negative_data['label'] = 0
data = pd.concat([data, negative_data]).reset_index(drop=True)

# Normalize text columns
data['ocr'] = data['ocr'].str.lower().str.strip()
data['claim'] = data['claim'].str.lower().str.strip()

# Generate embeddings using Sentence Transformers
print("Generating sentence embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a pretrained model
ocr_embeddings = model.encode(data['ocr'].tolist(), show_progress_bar=True)
claim_embeddings = model.encode(data['claim'].tolist(), show_progress_bar=True)

# Combine embeddings
X = np.hstack([
    ocr_embeddings,  # Include OCR embeddings
    claim_embeddings,  # Include Claim embeddings
    np.abs(ocr_embeddings - claim_embeddings),  # Element-wise difference
])
y = data['label'].values

# Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression with GridSearchCV
log_reg = LogisticRegression()
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [200, 500, 1000]
}

print("Running GridSearchCV for Logistic Regression...")
grid_search = GridSearchCV(
    log_reg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=3
)
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Predictions and Evaluation
y_test_pred = grid_search.best_estimator_.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

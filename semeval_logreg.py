import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

# Filter for low-similarity pairs as better negative examples
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
negative_data['similarity'] = negative_data.apply(lambda x: cosine_similarity(
    tfidf_vectorizer.fit_transform([x['ocr'], x['claim']])
)[0, 1], axis=1)
negative_data = negative_data[negative_data['similarity'] < 0.3]

# Combine datasets
data = pd.concat([data, negative_data.drop(columns=['similarity'])]).reset_index(drop=True)

# Normalize text columns
data['ocr'] = data['ocr'].str.lower().str.strip()
data['claim'] = data['claim'].str.lower().str.strip()

# Feature Engineering
print("Generating TF-IDF features...")
ocr_tfidf = tfidf_vectorizer.fit_transform(data['ocr'])
claim_tfidf = tfidf_vectorizer.transform(data['claim'])

# Cosine Similarity
data['cosine_similarity'] = [
    cosine_similarity(ocr_tfidf[i], claim_tfidf[i])[0][0] for i in range(len(data))
]

# Length-Based Features
data['length_diff'] = abs(data['ocr'].str.len() - data['claim'].str.len())

# Prepare Features and Labels
X = data[['cosine_similarity', 'length_diff']].values
y = data['label'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression with GridSearchCV
log_reg = LogisticRegression()

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.0005, 0.001, 0.005],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced'],
    'l1_ratio': [0.1, 0.5, 0.9],
    'max_iter': [200, 300, 500, 1000]
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

# Best parameters and model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
best_log_reg = grid_search.best_estimator_

# Predictions and Evaluation
y_test_pred_probs = best_log_reg.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_probs > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Not Relevant', 'Relevant'])
plt.yticks([0, 1], ['Not Relevant', 'Relevant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# Visualization: Precision-Recall Curve
disp = PrecisionRecallDisplay.from_predictions(y_test, y_test_pred_probs)
disp.ax_.set_title('Precision-Recall Curve')
plt.show()

# Visualization: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

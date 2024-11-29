import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load the CSV files
posts_df = pd.read_csv('C:/Users/kiara/OneDrive/Documents/NLP/243/Final Project/posts.csv')
fact_checks_df = pd.read_csv('C:/Users/kiara/OneDrive/Documents/NLP/243/Final Project/fact_checks.csv')
pairs_df = pd.read_csv('C:/Users/kiara/OneDrive/Documents/NLP/243/Final Project/pairs.csv')

# Create positive examples by merging pairs_df with posts_df and fact_checks_df
data = pd.merge(
    pd.merge(pairs_df, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id')
data['label'] = 1

# Create the 'features' column by concatenating 'ocr' and 'claim'
data['features'] = data['ocr'] + " " + data['claim']

# Generate negative examples by pairing posts with random fact checks
negative_pairs = pairs_df.copy()
negative_pairs['fact_check_id'] = negative_pairs['fact_check_id'].sample(frac=1, random_state=42).reset_index(drop=True)
negative_data = pd.merge(
    pd.merge(negative_pairs, posts_df[['post_id', 'ocr']], on='post_id'),
    fact_checks_df[['fact_check_id', 'claim']], on='fact_check_id'
)
negative_data['label'] = 0

# Combine positive and negative examples
data = pd.concat([data, negative_data]).reset_index(drop=True)

# Normalize text columns
data['ocr'] = data['ocr'].str.lower().str.strip()
data['claim'] = data['claim'].str.lower().str.strip()

# Feature Engineering

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
ocr_tfidf = tfidf_vectorizer.fit_transform(data['ocr'])
claim_tfidf = tfidf_vectorizer.transform(data['claim'])

# Cosine Similarity
data['cosine_similarity'] = [
    cosine_similarity(ocr_tfidf[i], claim_tfidf[i])[0][0] for i in range(len(data))
]

# Length-Based Features
data['ocr_length'] = data['ocr'].str.len()
data['claim_length'] = data['claim'].str.len()
data['length_diff'] = abs(data['ocr_length'] - data['claim_length'])

# N-Gram Overlap
count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
ocr_ngrams = count_vectorizer.fit_transform(data['ocr'])
claim_ngrams = count_vectorizer.transform(data['claim'])

data['ngram_overlap'] = [
    len(set(ocr_ngrams[i].indices) & set(claim_ngrams[i].indices)) for i in range(len(data))
]

# Jaccard Similarity
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

data['jaccard_similarity'] = data.apply(lambda x: jaccard_similarity(x['ocr'], x['claim']), axis=1)

# Prepare Features and Labels
X = data[['cosine_similarity', 'length_diff', 'ngram_overlap', 'jaccard_similarity']].values
y = data['label'].values

# Scale features for better convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, data[['post_id', 'fact_check_id']], test_size=0.2, random_state=42
)

# Logistic Regression with increased iterations
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Predictions
y_test_pred_probs = log_reg.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_probs > 0.5).astype(int)

# Evaluate Model
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Save Relevant Fact Check Predictions
test_results = test_ids.copy()
test_results['predicted_label'] = y_test_pred
relevant_fact_checks = (
    test_results[test_results['predicted_label'] == 1]
    .groupby('post_id')['fact_check_id']
    .apply(list)
    .reset_index()
)
relevant_fact_checks.to_csv('C:/Users/kiara/OneDrive/Documents/NLP/243/Final Project/monolingual_predictions.csv', index=False)

print("Predictions saved to monolingual_predictions.csv.")

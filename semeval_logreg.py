import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import json

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
data['features'] = data['features'].str.lower().str.strip()

# Vectorize the 'ocr' and 'claim' columns using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 terms
ocr_tfidf = tfidf_vectorizer.fit_transform(data['ocr'])
claim_tfidf = tfidf_vectorizer.transform(data['claim'])

# Compute cosine similarity between 'ocr' and 'claim'
data['cosine_similarity'] = [
    cosine_similarity(ocr_tfidf[i], claim_tfidf[i])[0][0] for i in range(len(data))
]

# Prepare features and labels
X = data[['cosine_similarity']].values  # Use cosine similarity as the only feature
y = data['label'].values  # Labels: 1 (relevant), 0 (not relevant)

# Split into train and test sets
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, data[['post_id', 'fact_check_id']], test_size=0.2, random_state=42
)

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict probabilities on the test set
y_test_pred_probs = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1 (relevant)

# Add predictions to test set
test_results = test_ids.copy()
test_results['predicted_label'] = (y_test_pred_probs > 0.5).astype(int)  # Threshold at 0.5
test_results['probability'] = y_test_pred_probs

# Filter relevant fact_check_id predictions (label == 1) for each post_id
relevant_fact_checks = (
    test_results[test_results['predicted_label'] == 1]
    .groupby('post_id')['fact_check_id']
    .apply(list)
    .reset_index()
)

# Convert to dictionary for JSON output
relevant_fact_checks_dict = relevant_fact_checks.set_index('post_id')['fact_check_id'].to_dict()

# Save the predictions as monolingual_predictions.json
with open('C:/Users/kiara/OneDrive/Documents/NLP/243/Final Project/monolingual_predictions.json', 'w') as f:
    json.dump(relevant_fact_checks_dict, f)

print("Relevant fact_check_ids for each post_id have been saved to 'monolingual_predictions.json'.")

# Load ground truth from pairs_df
ground_truth = pairs_df.groupby('post_id')['fact_check_id'].apply(list).to_dict()

# Define a function to calculate Top-k Accuracy
def calculate_top_k_accuracy(ground_truth_dict, predictions_dict, k=5):
    total_posts = len(ground_truth_dict)
    correct_predictions = 0

    for post_id, true_fact_checks in ground_truth_dict.items():
        predicted_fact_checks = predictions_dict.get(post_id, [])[:k]  # Get top-k predictions
        if set(predicted_fact_checks) & set(true_fact_checks):  # Check for overlap
            correct_predictions += 1

    top_k_accuracy = correct_predictions / total_posts
    return top_k_accuracy

# Calculate Top-k Accuracy
k = 5
top_k_accuracy = calculate_top_k_accuracy(ground_truth, relevant_fact_checks_dict, k=k)

print(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")

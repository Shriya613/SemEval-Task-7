# -*- coding: utf-8 -*-
"""Copy of NLP243SemEvalVisF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BW9v9SP5KXNsNUMvT2CBkSq9W7Rfxc2W
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

# Load the CSV files
posts_df = pd.read_csv('/content/posts.csv')
fact_checks_df = pd.read_csv('/content/fact_checks.csv')
pairs_df = pd.read_csv('/content/pairs.csv')

# Inspect the posts data
print("Posts DataFrame Info:")
print(posts_df.info())  # Get data types and non-null counts
print("\nFirst few rows of posts data:")
print(posts_df.head())  # Preview the first few rows

# Inspect the fact-checks data
print("\nFact-Checks DataFrame Info:")
print(fact_checks_df.info())  # Get data types and non-null counts
print("\nFirst few rows of fact-checks data:")
print(fact_checks_df.head())  # Preview the first few rows

# Inspect the pairs data
print("\nPairs DataFrame Info:")
print(pairs_df.info())  # Get data types and non-null counts
print("\nFirst few rows of pairs data:")
print(pairs_df.head())  # Preview the first few rows

# Calculate missing values in each DataFrame
print("Posts DataFrame Missing Values:\n", posts_df.isnull().sum())
print("Fact-Checks DataFrame Missing Values:\n", fact_checks_df.isnull().sum())
print("Pairs DataFrame Missing Values:\n", pairs_df.isnull().sum())

# Install FastText and download the pre-trained model if needed
#!pip install fasttext
#!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


# Load the pre-trained model for language detection
ft_lang_model = fasttext.load_model("lid.176.bin")

# Function to detect language using fastText
def detect_language_fasttext(text):
    if pd.notnull(text) and len(text) > 3:  # Filter out texts with fewer than 4 characters
        lang = ft_lang_model.predict(text)[0][0].split("__label__")[1]
        return lang
    else:
        return 'unknown'  # Assign 'unknown' for short or null texts

# Apply to both posts_df and fact_checks_df
posts_df['post_lang'] = posts_df['text'].apply(detect_language_fasttext)
fact_checks_df['fact_check_lang'] = fact_checks_df['claim'].apply(detect_language_fasttext)

# Merge pairs_df with the language information from posts_df and fact_checks_df
merged_df = pairs_df \
    .merge(posts_df[['post_id', 'post_lang']], on='post_id', how='left') \
    .merge(fact_checks_df[['fact_check_id', 'fact_check_lang']], on='fact_check_id', how='left')

# Create a `pair_lang` column to show the language pairing between SMP and FC
merged_df['pair_lang'] = merged_df['post_lang'] + '-' + merged_df['fact_check_lang']

# Count the occurrences of each language pair
pair_counts = merged_df['pair_lang'].value_counts().reset_index()
pair_counts.columns = ['pair_lang', 'count']

# Print the counts of language pairs
print("Counts of Language Pairs:")
print(pair_counts.sort_values(by='count', ascending=False).reset_index(drop=True))

# Visualization: Top 10 Language Pairs
top_pairs = pair_counts.nlargest(10, 'count')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_pairs, x='pair_lang', y='count', palette='pastel')
plt.title("Top 10 Language Pairs Count", fontsize=16)
plt.xlabel("Language Pair", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()
# This visualization helps identify which language pairs are most frequent, assisting in understanding the multilingual landscape of the dataset.

# Posts and Fact-Checks Text Length Analysis
posts_df['text_length'] = posts_df['text'].fillna('').apply(len)
fact_checks_df['claim_length'] = fact_checks_df['claim'].fillna('').apply(len)

# Visualization: Distribution of Text Lengths in Posts and Fact-Checks
plt.figure(figsize=(12, 5))
sns.histplot(posts_df['text_length'], kde=True, color="blue", label="Posts Text Length")
sns.histplot(fact_checks_df['claim_length'], kde=True, color="orange", label="Fact-Check Claim Length")
plt.xlabel("Text Length")
plt.title("Distribution of Text Lengths in Posts and Fact-Checks")
plt.legend()
plt.grid(axis='y')
plt.show()
# This histogram allows comparison of the lengths of posts and fact-check claims, indicating the verbosity of each type and possible implications for processing and model training.

# Verdicts Frequency Analysis
posts_df['verdicts_list'] = posts_df['verdicts'].apply(eval)
verdicts_flat = list(chain.from_iterable(posts_df['verdicts_list']))

# Visualization: Frequency of Verdicts in Posts
pink_palette = ["#ffc0cb", "#f08080", "#d3545b", "#e0ffff"]
plt.figure(figsize=(10, 6))
sns.countplot(y=verdicts_flat, order=pd.Series(verdicts_flat).value_counts().index, palette=pink_palette)
plt.title("Frequency of Verdicts in Posts")
plt.xlabel("Count")
plt.ylabel("Verdict Type")
plt.grid(axis='x')
plt.show()
# This count plot shows the distribution of different verdict types in the posts, helping to identify prevalent categories and trends in the data.

# Platform Type Analysis
posts_df['platforms'] = posts_df['instances'].apply(lambda x: [instance[1] for instance in eval(x)] if isinstance(x, str) else [])
platform_counts = posts_df.explode('platforms')['platforms'].value_counts()

# Visualization: Distribution of Post Platforms
plt.figure(figsize=(10, 6))
sns.barplot(x=platform_counts.values, y=platform_counts.index, palette="pastel")
plt.title("Distribution of Post Platforms")
plt.xlabel("Count")
plt.ylabel("Platform Type")
plt.grid(axis='y')
plt.show()
# This bar plot shows the distribution of posts across different platforms, providing insight into where fact-checking is most relevant.

# Title Presence Analysis
title_counts = fact_checks_df['title'].notnull().value_counts()

# Visualization: Pie Chart for Fact-Checks with vs. Without Titles
plt.figure(figsize=(8, 6))
plt.pie(title_counts, labels=["With Title", "Without Title"], autopct='%1.1f%%', startangle=140, colors=pink_palette)
plt.title("Fact-Checks with vs. Without Titles")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
# This pie chart illustrates the proportion of fact-checks that have titles versus those that do not, helping assess the completeness of the dataset.

# Select only numeric columns for the correlation matrix
numeric_df = merged_df.select_dtypes(include=['number'])

# Check if numeric_df is empty
if numeric_df.empty:
    print("No numeric columns available for correlation matrix.")
else:
    plt.figure(figsize=(10, 6))
    correlation_matrix = numeric_df.corr()  # Calculate the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

#!pip install wordcloud
#from wordcloud import WordCloud

# Combine text from both DataFrames
combined_text = ' '.join(posts_df['text'].dropna()) + ' ' + ' '.join(fact_checks_df['claim'].dropna())

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='flare').generate(combined_text)

# Plot the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title("Word Cloud of Posts and Claims")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Get the top 5 language pairs
top_pairs = pair_counts.nlargest(5, 'count')['pair_lang'].tolist()

# Set up the plot style
sns.set(style="whitegrid")

# Create individual plots for each top language pair
for pair in top_pairs:
    # Filter the dataframe for the current language pair
    filtered_df = merged_df[merged_df['pair_lang'] == pair]

    # Check if 'sentiment' exists in filtered_df
    if 'sentiment' in filtered_df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=filtered_df, x='sentiment', bins=10, color='pink', kde=True)  # Added kde for smoothness

        # Descriptive titles and labels
        plt.title(f"Sentiment Distribution for Language Pair: {pair}", fontsize=16)
        plt.xlabel("Sentiment Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # Display total counts in the plot
        total_counts = filtered_df['sentiment'].count()
        plt.text(0.8, 0.9, f'Total Observations: {total_counts}', fontsize=12, ha='center', transform=plt.gca().transAxes)

        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional grid for better readability
        plt.show()
    else:
        print(f"Sentiment column not found in filtered_df for language pair: {pair}.")

#pip install transformers

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained multilingual model (e.g., XLM-Roberta)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")

# Function to generate embeddings using XLM-Roberta
def generate_embeddings(texts):
    # Tokenize and encode the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get sentence-level embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Generate embeddings for post texts and fact-check claims
post_texts = posts_df['text'].fillna('').tolist()
claim_texts = fact_checks_df['claim'].fillna('').tolist()

post_embeddings = generate_embeddings(post_texts)
claim_embeddings = generate_embeddings(claim_texts)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(post_embeddings, claim_embeddings)

# Retrieve top-k most similar fact-checks for each post
top_k = 3
top_k_matches = {}
for i, post_id in enumerate(posts['post_id']):
    # Get indices of the top-k most similar claims
    top_indices = similarity_matrix[i].argsort()[-top_k:][::-1]
    top_fact_check_ids = [fact_checks.iloc[idx]['fact_check_id'] for idx in top_indices]
    top_k_matches[post_id] = top_fact_check_ids

# Display results
for post_id, fact_check_ids in top_k_matches.items():
    print(f"Post ID: {post_id} -> Top {top_k} Fact-Check IDs: {fact_check_ids}")

# Display results
for post_id, fact_check_ids in top_k_matches.items():
    print(f"Post ID: {post_id} -> Top {top_k} Fact-Check IDs: {fact_check_ids}")
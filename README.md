README.md

## SemEval Task 7: Multilingual Fact-Checked Information Retrieval

### NLP 243 | Final Project | Group 1

Contributors: Ishika Kulkarni (ikulkar1@ucsc.edu), Shriya Sravani Y (sy4@ucsc.edu), Kiara LaRocca (klarocca@ucsc.edu)

This task aims to evaluate the effectiveness of systems in retrieving relevant fact-checks for multilingual social media content, emphasizing the need for scalable, efficient models that can operate across diverse linguistic contexts and assist fact-checkers worldwide. Our task, specifically, is to retrieve the top-K most relevant fact-check claims for each social media post in a multilingual setting. Given a post in one language and fact-checked claims in potentially different languages, the system calculates similarity scores to rank the most relevant claims. Although it seems like the task is the information retrieval itself, it is actually a binary classification task to determine whether a given fact-checked claim is relevant to the post, regardless of the fact-check verdict. In order to implement this task, we provide the dataset and the information necessary to run our different models.

___________________________________

## The Dataset

The dataset given to us includes the following files:

1. fact_checks.csv - contains a subset of 153743 fact-checks in 8 languages ('ara', 'deu', 'eng', 'fra', 'msa', 'por', 'spa', 'tha') covering all subtasks.

2. posts.csv - contains all monolingual train/dev posts and crosslingual train/dev posts (there is no overlap between the two subsets). It contains posts in 14 languages.

3. pairs.csv - contains all train pairs (monolingual and crosslingual)

4. tasks.json - JSON file containing a list of fact-check IDs, train posts IDs and dev post IDs for each of the subtasks (monolingual - for 8 languages, crosslingual)

5. monolingual_predictions.json - a submission file for monolingual task containing dev post IDs and an empty column expecting a list of retrieved fact-checks for each row (post ID)

In order to use this dataset, we created our own data file using specific columns from each CSV file (data.csv). While data.csv is available in the repository, some code files still read in data as individual files and concatenates them. Therefore, while available, set up may not include data.csv.

___________________________________

## Set-up for Logistic Regression (semeval_logreg.py)

To begin set-up for this file, follow these steps:
1. Make sure to install the required Python libraries.
2. Prepare the Dataset (posts.csv, fact_checks.csv, pairs.csv)
3. Save the file as a Python script and execute it.

The script does not include downloading the dataset, as it expects the dataset to be available in the directory. It does address dataset preparation, training, and evaluating the model.

The output of the script includes:
* Best hyperparameters and cross-validation accuracy
* Classification metrics (precision, recall, and F1-score)
* Overall accuracy and Top-K accuracy for the test set

___________________________________

## Set-up for roBERTa Models (v4, v5):

* The code expects the dataset in the same directory and requires the installation of a few Python libraries.
* Once the code runs, the expected output is a predictions.csv for both of them, where the models and text files containing the loss, accuracy, and other values will be saved.

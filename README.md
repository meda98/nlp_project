# NLP Sentiment Analysis Project
This repository contains a Natural Language Processing (NLP) project that performs sentiment analysis on movie reviews using classical machine learning techniques.

## Project Overview
The project uses the *IMDb Large Movie Review Dataset* to train and evaluate sentiment classification models. The workflow includes:
- Data loading and preprocessing
- Text cleaning and normalization
- TF-IDF feature extraction (unigrams and bigrams)
- Model training and evaluation
- Performance visualization
- Sentiment prediction on external review data

The following classifiers are compared:
- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Dataset Setup
This project requires the *Large Movie Review Dataset (IMDb)* by Maas et al.

Download the dataset from:

```console
https://ai.stanford.edu/~amaas/data/sentiment/
```

### Installation Steps
1. Download the dataset ZIP file from the link above.
2. Extract the ZIP file.
3. Move the extracted aclImdb folder into the project's root directory.

The project structure should look similar to:

```console
project/
│
├── aclImdb/
│   ├── train/
│   └── test/
│
├── nltk_data/
├── main.py
├── movie_sentiment.py
├── data_loading.py
├── data_preprocessing.py
├── evaluation.py
├── visualization.py
└── remarkably_bright_creatures_reviews.xlsx
```

## Additional Files
### nltk_data
The repository includes a local *nltk_data* directory containing the required NLTK resources used during preprocessing.

### Remarkably Bright Creatures Reviews
The file *remarkably_bright_creatures_reviews.xlsx* contains review data used for sentiment prediction with the final trained model.

## Running the Project
### Train and Evaluate Models
```console
python main.py
```

This script trains all models, evaluates them on different dataset sizes, and generates performance visualizations.

### Predict Sentiment for New Reviews
```console
python movie_sentiment.py
```

This script trains the final Logistic Regression model and predicts the sentiment of reviews stored in *remarkably_bright_creatures_reviews.xlsx*.

## Technologies
- Python
- NLTK
- Scikit-learn
- Pandas
- Matplotlib

import pandas as pd

from data_loading import load_imdb_subset
from data_preprocessing import preprocess_reviews

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("Loading IMDB dataset...")

# Load the full IMDB training dataset consisting of 12,500 positive and 12,500 negative reviews
X_train, y_train = load_imdb_subset(
    "aclImdb/train",
    12500,
    12500
)

print("Preprocessing reviews...")

# Apply preprocessing pipeline to all training reviews
X_train_clean = preprocess_reviews(X_train)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)

# Learn vocabulary from the training data and transform reviews into TF-IDF feature vectors
X_train_tfidf = vectorizer.fit_transform(X_train_clean)

# Train the best-performing classification model
print("Training Logistic Regression model...")

# Initialize Logistic Regression classifier
model = LogisticRegression(max_iter=1000)

# Train the classifier on the TF-IDF encoded reviews
model.fit(X_train_tfidf, y_train)

# ============================================================
# LOAD MOVIE REVIEWS
# ============================================================

print("Loading movie reviews...")

# Load unseen movie reviews from Excel file
reviews_df = pd.read_excel(
    "remarkably_bright_creatures_reviews.xlsx"
)

# Extract review texts from the "review" column
reviews = reviews_df["review"].tolist()

# ============================================================
# PREPROCESS NEW REVIEWS
# ============================================================

# Apply the same preprocessing pipeline used for the IMDB training data
reviews_clean = preprocess_reviews(reviews)

# Transform reviews into TF-IDF vectors using the vocabulary learned from the training data
X_movie = vectorizer.transform(reviews_clean)

# ============================================================
# PREDICT SENTIMENTS
# ============================================================

# Predict sentiment labels for all movie reviews
predictions = model.predict(X_movie)

# Convert numerical labels into readable sentiment classes
predicted_labels = [
    "Positive" if pred == 1 else "Negative"
    for pred in predictions
]

# Store predictions in the DataFrame
reviews_df["Predicted Sentiment"] = predicted_labels

# ============================================================
# DETERMINE OVERALL SENTIMENT
# ============================================================

# Count positive and negative predictions
positive_count = sum(predictions == 1)
negative_count = sum(predictions == 0)

# Calculate the proportion of positive reviews
positive_ratio = positive_count / len(predictions)

# Determine overall movie sentiment based on the majority of classified reviews
overall_sentiment = (
    "Positive"
    if positive_ratio >= 0.5
    else "Negative"
)

# ============================================================
# OUTPUT RESULTS
# ============================================================

print("\nIndividual Predictions")
print("=" * 60)

# Print prediction result for each review
for i, row in reviews_df.iterrows():
    print(f"\nReview {i+1}:")
    print(row["review"][:150] + "...")
    print("Prediction:", row["Predicted Sentiment"])

print("\n" + "=" * 60)
print("Overall Movie Sentiment")
print("=" * 60)

# Print summary statistics
print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Positive ratio: {positive_ratio:.2%}")

# Print final overall sentiment classification
print(f"\nOverall sentiment: {overall_sentiment}")
from data_loading import load_imdb_subset
from data_preprocessing import preprocess_reviews
from visualization import plot_metrics
from evaluation import compute_metrics, print_metrics, print_confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Store evaluation results for all experiments
results = []

def run_experiment(n_pos, n_neg):
    """
    Runs a sentiment classification experiment for a given dataset size.

    Args:
        n_pos: Number of positive reviews.
        n_neg: Number of negative reviews.

    Returns:
        None
    """

    print("\n" + "#" * 70)
    print(f"Dataset size: {n_pos + n_neg} training reviews")
    print("#" * 70)

    # Load data
    X_train, y_train = load_imdb_subset("aclImdb/train", n_pos, n_neg)
    X_test, y_test = load_imdb_subset("aclImdb/test", n_pos, n_neg)

    # Preprocess reviews
    X_train_clean = preprocess_reviews(X_train)
    X_test_clean = preprocess_reviews(X_test)

    #print(X_train_clean[1])

    # Initialize TF-IDF vectorizer (unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=10000,
        min_df=2
    )

    # Learn vocabulary from training data and transform training reviews into feature vectors
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)

    # Transform test reviews using the same vocabulary learned from training data
    X_test_tfidf = vectorizer.transform(X_test_clean)

    # Define models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC()
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print("=" * 60)
        print(model_name)
        print("=" * 60)

        # Train the classifier on the TF-IDF encoded training data
        model.fit(X_train_tfidf, y_train)
        
        # Predict sentiments for the test data
        y_pred = model.predict(X_test_tfidf)

        # Compute and print evaluation metrics
        scores = compute_metrics(y_test, y_pred)
        print_metrics(scores)

        results.append({
            "dataset_size": n_pos + n_neg,
            "model": model_name,
            **scores
        })

        # Print confusion matrix for the best-performing setup
        if model_name == "Logistic Regression" and n_pos == 12500:
            print_confusion_matrix(y_test, y_pred)

# Different dataset sizes
dataset_sizes = [
    (500, 500),
    (2500, 2500),
    (7500, 7500),
    (12500, 12500)
]

# Run experiments for all dataset sizes
for n_pos, n_neg in dataset_sizes:
    run_experiment(n_pos, n_neg)

# Visualize results across dataset sizes
plot_metrics(results)
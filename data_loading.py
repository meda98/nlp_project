import os
from sklearn.utils import shuffle

def load_reviews(folder_path: str, label: int, limit: int | None = None) -> tuple[list[str], list[int]]:
    """
    Loads review texts from a folder and assigns a label to each review.

    Args:
        folder_path: Path to the directory containing review .txt files.
        label: Class label assigned to each review (1 = positive, 0 = negative).
        limit: Optional maximum number of reviews to load. If None, all reviews are loaded.

    Returns:
        A tuple containing:
            - A list of review texts.
            - A list of corresponding labels.
    """

    reviews: list[str] = []
    labels: list[int] = []

    files = os.listdir(folder_path)

    if limit:
        files = files[:limit]

    for file in files:
        with open(os.path.join(folder_path, file), encoding="utf8") as f:
            text = f.read()
            reviews.append(text)
            labels.append(label)

    return reviews, labels


def load_imdb_subset(base_path: str, n_pos: int, n_neg: int) -> tuple[list[str], list[int]]:
    """
    Loads a subset of the IMDB dataset with a specified number of positive and negative reviews.

    Args:
        base_path: Path to the dataset split (e.g., 'aclImdb/train' or 'aclImdb/test').
        n_pos: Number of positive reviews to load.
        n_neg: Number of negative reviews to load.

    Returns:
        A shuffled dataset consisting of:
            - X: List of review texts.
            - y: List of corresponding labels (1 = positive, 0 = negative).
    """

    pos_path = os.path.join(base_path, "pos")
    neg_path = os.path.join(base_path, "neg")

    pos_reviews, pos_labels = load_reviews(pos_path, 1, n_pos)
    neg_reviews, neg_labels = load_reviews(neg_path, 0, n_neg)

    X = pos_reviews + neg_reviews
    y = pos_labels + neg_labels

    return shuffle(X, y, random_state=42)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Computes evaluation metrics for classification results.

    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
        
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Return metrics as dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def print_metrics(scores):
    """
    Prints evaluation metrics to the console.

    Args:
        scores: Dictionary containing evaluation metrics.

    Returns:
        None
    """
    
    print("Accuracy:", scores["accuracy"])
    print("Precision:", scores["precision"])
    print("Recall:", scores["recall"])
    print("F1-score:", scores["f1_score"])

def print_confusion_matrix(y_true, y_pred):
    """
    Prints the confusion matrix for classification results.

    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.

    Returns:
        None
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
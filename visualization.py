import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(results):
    """
    Visualizes evaluation metrics across different dataset sizes.

    Args:
        results: List of dictionaries containing dataset size, model name,
                 and evaluation metrics.

    Returns:
        None
    """
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)

    # Define metrics to plot
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Loop over metrics
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot one line per model
        for model_name in results_df["model"].unique():
            model_results = results_df[
                results_df["model"] == model_name
            ].sort_values("dataset_size")

            ax.plot(
                model_results["dataset_size"],
                model_results[metric],
                marker="o",
                label=model_name
            )

        # Set labels and title
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Dataset size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
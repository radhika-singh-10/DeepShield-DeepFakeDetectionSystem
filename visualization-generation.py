import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

# Load metrics from the pickle file
def load_metrics(file_path):
    with open(file_path, "rb") as f:
        metrics = pickle.load(f)
    return metrics

# Plot line graphs for loss and accuracy
def plot_loss_accuracy(training_losses, validation_losses, training_accuracies, validation_accuracies, output_path):
    plt.figure(figsize=(12, 6))

    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label="Training Loss", marker='o')
    plt.plot(validation_losses, label="Validation Loss", marker='o')
    plt.title("Loss Function Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label="Training Accuracy", marker='o')
    plt.plot(validation_accuracies, label="Validation Accuracy", marker='o')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Plot heatmap for precision, recall, and F1 scores
def plot_heatmap(precisions, recalls, f1_scores, output_path):
    metrics_array = np.array([precisions, recalls, f1_scores])
    metrics_labels = ["Precision", "Recall", "F1 Score"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_array, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=range(1, len(precisions) + 1), yticklabels=metrics_labels)
    plt.title("Precision, Recall, and F1 Score Heatmap")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.savefig(output_path)
    plt.show()

# Main function to generate the plots
def main():
    metrics_file = "deepstack_final_training_metrics.pkl"
    metrics = load_metrics(metrics_file)

    training_losses = metrics["training_losses"]
    training_accuracies = metrics["training_accuracies"]
    validation_losses = metrics["validation_losses"]
    validation_accuracies = metrics["validation_accuracies"]
    precisions = metrics["precisions"]
    recalls = metrics["recalls"]
    f1_scores = metrics["f1_scores"]

    # Generate and save line plots for loss and accuracy
    plot_loss_accuracy(
        training_losses, validation_losses,
        training_accuracies, validation_accuracies,
        "loss_accuracy_graph.png"
    )

    # Generate and save heatmap for precision, recall, and F1 scores
    plot_heatmap(precisions, recalls, f1_scores, "precision_recall_f1_heatmap.png")

if __name__ == "__main__":
    main()

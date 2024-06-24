import argparse
import os
import csv
import sys
import numpy as np
import joblib
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import modified_get_embedding

csv.field_size_limit(sys.maxsize)

# This script creates and trains a KNN classifier
# And can be used as an evaluation step of the DNABERT model and is modified to work for only one dataset instead of multiple

def save_embeddings(embeddings, labels, filename):
    # Save embeddings and their corresponding labels to a file using joblib for later use
    data = {'embeddings': embeddings, 'labels': labels}
    joblib.dump(data, filename)

# Function to train the KNN classifier
def train_knn_classifier(embedding_file, n_neighbors=3):
    # Load embeddings and labels from the saved file
    data = joblib.load(embedding_file)
    embeddings = data['embeddings']
    labels = data['labels']
    # Initialize and train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(embeddings, labels)
    return knn

# Function to classify a given DNA sequence using the trained KNN classifier
def classify_sequence(sequence, knn_model, model, species, sample, test_model_dir):
    # Generate an embedding for the sequence
    embedding = modified_get_embedding([sequence], model, species, sample, test_model_dir=test_model_dir)
    # Standardize the embedding
    embedding_standard = StandardScaler().fit_transform(embedding)
    # Predict the class using the KNN classifier
    prediction = knn_model.predict(embedding_standard)
    return prediction

def main(args):
    # Split the model list from a comma-separated string into a list
    model_list = args.model_list.split(",")
    for model in model_list:
        species = "reference"
        max_length = 10000
        sample = "0"

        print(f"Start {model} {species} clustering")
        # Path to the evaluation data file
        data_file = os.path.join(args.data_dir, species, "evaluation_sequences.tsv")

        # Read the evaluation data from a TSV file
        with open(data_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            data = list(reader)[1:]  # Skip the header row

        # Extract DNA sequences and labels from the data
        dna_sequences = [d[0][:max_length] for d in data]
        labels = [d[1] for d in data]

        # Convert labels to numeric values
        label2id = {l: i for i, l in enumerate(set(labels))}
        labels = np.array([label2id[l] for l in labels])
        num_clusters = len(label2id)
        print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

        # Generate embeddings for the DNA sequences
        try:
            print(f"Loading model from directory: {args.test_model_dir}")
            # Debug: List all files in the model directory to ensure correct path
            for root, dirs, files in os.walk(args.test_model_dir):
                for file in files:
                    print(os.path.join(root, file))

            # Generate embeddings using the provided model
            embedding = modified_get_embedding(dna_sequences, model, species, sample, test_model_dir=args.test_model_dir)
            # Normalize and standardize the embeddings
            embedding_norm = normalize(embedding)
            embedding_standard = StandardScaler().fit_transform(embedding_norm)

            # Save the embeddings and labels to a file
            embedding_filename = f'embeddings_{model}_{species}_{sample}.pkl'
            save_embeddings(embedding_standard, labels, embedding_filename)
            print(f"Embeddings saved to {embedding_filename}")

            # Train the KNN classifier using the saved embeddings
            knn_classifier = train_knn_classifier(embedding_filename)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(embedding_standard, labels, test_size=0.2, random_state=42)

            print("Shapes:")
            print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
            print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

            # Predict the labels for the test set
            y_pred = knn_classifier.predict(X_test)

            print("y_test sample:", y_test[:10])
            print("y_pred sample:", y_pred[:10])

            # Check if y_test and y_pred are identical (this would be unusual)
            if np.array_equal(y_test, y_pred):
                print("Warning: y_test and y_pred are identical, which is unexpected.")

            # Calculate accuracy and generate a classification report
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            print("Accuracy:", accuracy)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(cm)

            # Calculate average precision and recall
            avg_precision = np.mean([v['precision'] for k, v in report.items() if isinstance(v, dict)])
            avg_recall = np.mean([v['recall'] for k, v in report.items() if isinstance(v, dict)])

            print("Average Precision:", avg_precision)
            print("Average Recall:", avg_recall)

            # Display confusion matrix for a subset of classes
            subset_classes = 10  # Number of classes to display in the confusion matrix
            unique_labels = np.unique(y_test)
            subset_labels = unique_labels[:subset_classes]
            mask = np.isin(y_test, subset_labels)
            cm_subset = confusion_matrix(y_test[mask], y_pred[mask], labels=subset_labels)

            # Map numeric labels back to original labels for display
            display_labels = [k for k, v in label2id.items() if v in subset_labels]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_subset, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.show()

        except Exception as e:
            print(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", type=str, required=True, help="Comma-separated list of models to evaluate")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--test_model_dir", type=str, required=True, help="Directory containing the model files")
    args = parser.parse_args()
    main(args)

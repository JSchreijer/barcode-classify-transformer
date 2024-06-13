import argparse
import os
import csv
import sys
import numpy as np
import joblib
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import modified_get_embedding

# This script evaluates a DNA sequence classification model using a linear classifier (Logistic Regression).
# The process involves generating embeddings for sequences, training a linear classifier, and evaluating its performance.
# This approach is often referred to as 'linear probing' because it assesses the quality of the embeddings by how well a linear model can classify them.

csv.field_size_limit(sys.maxsize)

def save_embeddings(embeddings, labels, filename):
    """
    Save embeddings and their corresponding labels to a file.

    Args:
        embeddings (numpy.ndarray): The embeddings of the sequences.
        labels (numpy.ndarray): The labels corresponding to each embedding.
        filename (str): The file where the embeddings and labels will be saved.
    """
    data = {'embeddings': embeddings, 'labels': labels}
    joblib.dump(data, filename)

def train_linear_classifier(embedding_file):
    """
    Train a linear classifier (Logistic Regression) on the embeddings.

    Args:
        embedding_file (str): The file containing the embeddings and labels.

    Returns:
        LogisticRegression: The trained classifier.
    """
    data = joblib.load(embedding_file)
    embeddings = data['embeddings']
    labels = data['labels']
    clf = LogisticRegression(random_state=42)
    clf.fit(embeddings, labels)
    return clf

def classify_sequence(sequence, clf, model, species, sample, test_model_dir):
    """
    Classify a given DNA sequence using a trained classifier.

    Args:
        sequence (str): The DNA sequence to classify.
        clf (LogisticRegression): The trained classifier.
        model (str): The model used for generating embeddings.
        species (str): The species category for the sequence.
        sample (str): The sample identifier.
        test_model_dir (str): The directory containing the model.

    Returns:
        numpy.ndarray: The predicted label for the sequence.
    """
    embedding = modified_get_embedding([sequence], model, species, sample, test_model_dir=test_model_dir)
    embedding_standard = StandardScaler().fit_transform(embedding)
    prediction = clf.predict(embedding_standard)
    return prediction

def main(args):
    """
    Main function to evaluate the model on DNA sequences.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # List of models to evaluate, provided as a comma-separated string
    model_list = args.model_list.split(",")
    for model in model_list:
        species = "reference"
        max_length = 10000
        sample = "0"

#This loads in the testing dataset which is the seperate 20% that includes Lena's files as well.
        #This file will be used to train the linear probing
        print(f"Start {model} {species} clustering")
        data_file = os.path.join(args.data_dir, species, "evaluation_sequences.tsv")

        # Read the data from the evaluation_sequences.tsv file
        with open(data_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            data = list(reader)[1:]  # Skip the header row

        dna_sequences = [d[0][:max_length] for d in data]
        labels = [d[1] for d in data]

        # Convert labels to numeric values
        label2id = {l: i for i, l in enumerate(set(labels))}
        labels = np.array([label2id[l] for l in labels])
        num_clusters = len(label2id)
        print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

        try:
            print(f"Loading model from directory: {args.test_model_dir}")
            for root, dirs, files in os.walk(args.test_model_dir):
                for file in files:
                    print(os.path.join(root, file))

            # Generate embeddings for the DNA sequences
            #normalizing and standardizing the embeddings afterwards
            embedding = modified_get_embedding(dna_sequences, model, species, sample, test_model_dir=args.test_model_dir)
            embedding_norm = normalize(embedding)
            embedding_standard = StandardScaler().fit_transform(embedding_norm)

            embedding_filename = f'embeddings_{model}_{species}_{sample}.pkl'
            save_embeddings(embedding_standard, labels, embedding_filename)
            print(f"Embeddings saved to {embedding_filename}")

            # Train linear classifier
            clf = train_linear_classifier(embedding_filename)

            # Evaluate accuracy using a train-test split
            X_train, X_test, y_train, y_test = train_test_split(embedding_standard, labels, test_size=0.2, random_state=42)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print("Accuracy:", accuracy)
            print("Classification Report:")
            print(report)

        except Exception as e:
            print(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DNA sequence classification model using linear probing.")
    parser.add_argument("--model_list", type=str, required=True, help="Comma-separated list of models to evaluate.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files.")
    parser.add_argument("--test_model_dir", type=str, required=True, help="Directory containing the model files.")
    args = parser.parse_args()
    main(args)

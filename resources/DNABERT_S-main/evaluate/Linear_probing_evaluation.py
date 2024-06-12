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

# TODO: please write a brief statement of what this script does here. Why is this called 'linear probing'?

csv.field_size_limit(sys.maxsize)

def save_embeddings(embeddings, labels, filename):
    # TODO: explain what this does as a docstring, including a description of the input arguments
    data = {'embeddings': embeddings, 'labels': labels}
    joblib.dump(data, filename)

def train_linear_classifier(embedding_file):
    # TODO: explain what this does as a docstring, including a description of the input and output
    data = joblib.load(embedding_file)
    embeddings = data['embeddings']
    labels = data['labels']
    clf = LogisticRegression(random_state=42)
    clf.fit(embeddings, labels)
    return clf

def classify_sequence(sequence, clf, model, species, sample, test_model_dir):
    # TODO: explain what this does as a docstring, including a description of the inputs and output
    embedding = modified_get_embedding([sequence], model, species, sample, test_model_dir=test_model_dir)
    embedding_standard = StandardScaler().fit_transform(embedding)
    prediction = clf.predict(embedding_standard)
    return prediction

def main(args):

    # TODO: what sort of list is this going to be? What values have you tested?
    model_list = args.model_list.split(",")
    for model in model_list:
        species = "reference"
        max_length = 10000
        sample = "0"

        # TODO: explain what this file is supposed to be
        print(f"Start {model} {species} clustering")
        data_file = os.path.join(args.data_dir, species, "evaluation_sequences.tsv")

        with open(data_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            data = list(reader)[1:] # TODO: why is this?

        dna_sequences = [d[0][:max_length] for d in data]
        labels = [d[1] for d in data]

        # Convert labels to numeric values
        # TODO: what happens here exactly? First a dict is made, what are the keys and values? What is the np.array?
        label2id = {l: i for i, l in enumerate(set(labels))}
        labels = np.array([label2id[l] for l in labels])
        num_clusters = len(label2id)
        print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

        # Generate embedding
        try:
            print(f"Loading model from directory: {args.test_model_dir}")
            # Debug: List all files in the model directory
            for root, dirs, files in os.walk(args.test_model_dir):
                for file in files:
                    print(os.path.join(root, file))

            # TODO: what happens here? What, if anything, is done with the normalized embedding?
            embedding = modified_get_embedding(dna_sequences, model, species, sample, test_model_dir=args.test_model_dir)
            embedding_norm = normalize(embedding)
            embedding_standard = StandardScaler().fit_transform(embedding)

            embedding_filename = f'embeddings_{model}_{species}_{sample}.pkl'
            save_embeddings(embedding_standard, labels, embedding_filename)
            print(f"Embeddings saved to {embedding_filename}")

            # Train linear classifier
            clf = train_linear_classifier(embedding_filename)

            # Evaluate accuracy
            # TODO: what happens here? Aren't the embeddings obtained from the entire data set (line 73)?
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
    parser = argparse.ArgumentParser()
    # TODO: Give help texts for what these arguments are. If possible, define defaults.
    parser.add_argument("--model_list", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_model_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

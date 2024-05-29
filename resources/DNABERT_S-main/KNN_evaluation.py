import os
from sklearn.preprocessing import normalize, StandardScaler
import csv
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from utils import get_embedding

csv.field_size_limit(sys.maxsize)


def save_embeddings(embeddings, labels, filename):
    data = {'embeddings': embeddings, 'labels': labels}
    joblib.dump(data, filename)


def train_knn_classifier(embedding_file, n_neighbors=5):
    data = joblib.load(embedding_file)
    embeddings = data['embeddings']
    labels = data['labels']
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(embeddings, labels)
    return knn


def classify_sequence(sequence, knn_model, model, species, sample, test_model_dir):
    embedding = get_embedding([sequence], model, species, sample, test_model_dir=test_model_dir)
    embedding_standard = StandardScaler().fit_transform(embedding)
    prediction = knn_model.predict(embedding_standard)
    return prediction


def main():
    # Fill in your data here
    model ="epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs6.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue"   # Name of the single model you are using
    data_dir = "/DNABERT_S-main/evaluate"  # Path to your data directory
    test_model_dir = "/DNABERT_S-main/pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs6.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best"  # Path to your test model directory

    for species in ["reference", "marine", "plant"]:
        max_length = 10000 if species == "reference" else 20000
        for sample in [0, 1, 2, 3, 4]:
            if species == "reference" and sample > 1:
                continue
            sample = str(sample)

            print(f"Start {model} {species} {sample} clustering")
            data_file = os.path.join(data_dir, species, f"clustering_{sample}.tsv")

            with open(data_file, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                data = list(reader)[1:]

            dna_sequences = [d[0][:max_length] for d in data]
            labels = [d[1] for d in data]

            # Convert labels to numeric values
            label2id = {l: i for i, l in enumerate(set(labels))}
            labels = np.array([label2id[l] for l in labels])
            num_clusters = len(label2id)
            print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

            # Generate embedding
            embedding = get_embedding(dna_sequences, model, species, sample, test_model_dir=test_model_dir)
            embedding_norm = normalize(embedding)
            embedding_standard = StandardScaler().fit_transform(embedding)

            embedding_filename = f'embeddings_{model}_{species}_{sample}.pkl'
            save_embeddings(embedding_standard, labels, embedding_filename)
            print(f"Embeddings saved to {embedding_filename}")


# Run the main function
main()

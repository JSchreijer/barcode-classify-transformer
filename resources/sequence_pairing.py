import os
import csv
import random
from Bio import SeqIO

# Define the input folder containing sequence files
input_folder = '../data/l0.2_s3_4_1500_o1.0_a0_constr_localpair/chunks/unaligned'

# Define the output CSV file
output_file = 'aligned_sequences.csv'

# Define the maximum number of sequence pairs per chunk
max_pairs_per_chunk = 500

def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file."""
    sequences = []
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))
    return sequences

def generate_pairs(items, num_pairs):
    """Generate unique pairs of sequences."""
    pairs = []
    while len(pairs) < num_pairs:
        # Shuffle the list of items
        random.shuffle(items)
        # Pair up the items
        for i in range(0, len(items) - 1, 2):
            pairs.append((items[i], items[i+1]))
            if len(pairs) == num_pairs:
                break
    return pairs[:num_pairs]

# Open the output CSV file in write mode
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Read sequences from the current file
        sequences = read_fasta_sequences(file_path)

        # Generate pairs of sequences
        pairs = generate_pairs(sequences, max_pairs_per_chunk)

        # Write pairs to the CSV file
        for pair in pairs:
            writer.writerow(pair)

print("Pairs have been successfully written to", output_file)

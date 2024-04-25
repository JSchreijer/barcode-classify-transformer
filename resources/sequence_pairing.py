
import os
import csv
import random
import argparse
from Bio import SeqIO
import pandas as pd

# Function to read FASTA sequences from a file using Biopython
def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file."""
    sequences = []
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))
    return sequences

# Function to generate pairs of sequences
def generate_pairs(items, num_pairs):
    pairs = []
    while len(pairs) < num_pairs:
        random.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            pairs.append((items[i], items[i + 1]))
            if len(pairs) == num_pairs:
                break
    return pairs[:num_pairs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pairs of sequences from FASTA files.")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing sequence files.")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument("--max_pairs_per_chunk", type=int, default=500, help="Maximum number of sequence pairs per chunk.")
    parser.add_argument("--excel_file", type=str, help="Path to the Excel file containing specific IDs to skip.")
    args = parser.parse_args()

    # Read the specific IDs from the Excel file
    df = pd.read_excel(args.excel_file)
    specific_ids = df['ID'].tolist()

    # Open the output CSV file in write mode
    with open(args.output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Loop through each file in the input folder
        for filename in os.listdir(args.input_folder):
            file_path = os.path.join(args.input_folder, filename)

            # Check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                # Read sequences from the current file using Biopython
                sequences = read_fasta_sequences(file_path)

                # Remove sequences with specific IDs
                sequences = [seq for seq in sequences if seq.split('|')[0] not in specific_ids]

                # Generate pairs of sequences, limited to max_pairs_per_chunk
                pairs = generate_pairs(sequences, args.max_pairs_per_chunk)

                # Write pairs to the CSV file
                for pair in pairs:
                    writer.writerow(pair)



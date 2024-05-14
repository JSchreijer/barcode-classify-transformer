import os
import csv
import random
import argparse
import pandas as pd
from Bio import SeqIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main(input_folder, output_file, excel_file, max_pairs_percentage):
    # Read the specific IDs from the Excel file
    df = pd.read_excel(excel_file)
    specific_ids = set(df['ID'].tolist())

    # Calculate total number of sequence pairs
    total_pairs = 0
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            sequences = read_fasta_sequences(file_path)
            total_pairs += len(sequences) // 2

    # Calculate the number of pairs to generate (80% of total)
    num_pairs = int(total_pairs * max_pairs_percentage)

    # Open the output CSV file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # List to store all sequences
        all_sequences = []

        # Loop through each file in the input folder
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            # Check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                logging.info(f'Reading sequences from file: {file_path}')

                # Read sequences from the current file using Biopython
                sequences = read_fasta_sequences(file_path)

                # Skip sequences with specific IDs
                sequences_without_specific_ids = [seq for seq in sequences if seq.split('|')[0] not in specific_ids]

                # Add sequences to the list of all sequences
                all_sequences.extend(sequences_without_specific_ids)

        # Select 80% of the total sequences
        selected_sequences = random.sample(all_sequences, num_pairs * 2)

        # Generate pairs of sequences
        pairs = generate_pairs(selected_sequences, num_pairs)
        logging.info(f'Generated {len(pairs)} pairs of sequences')

        # Write pairs to the CSV file
        for pair in pairs:
            writer.writerow(pair)

        logging.info('Script execution completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pairs of sequences from a folder of sequence files.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing sequence files.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument("excel_file", type=str, help="Path to the Excel file containing specific IDs.")
    parser.add_argument("--max_pairs_percentage", type=float, default=0.8, help="Maximum percentage of pairs to generate (default: 0.8).")
    args = parser.parse_args()

    main(args.input_folder, args.output_file, args.excel_file, args.max_pairs_percentage)

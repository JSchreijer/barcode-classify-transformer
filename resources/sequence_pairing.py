import os
import csv
import random
from Bio import SeqIO
import pandas as pd
import logging
import argparse

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

def main(args):
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
                logging.info(f'Reading sequences from file: {file_path}')

                # Read sequences from the current file using Biopython
                sequences = read_fasta_sequences(file_path)

                # Select sequences without specific IDs
                sequences_without_specific_ids = [seq for seq in sequences if seq.split('|')[0] not in specific_ids]
                logging.info(f'Found {len(sequences_without_specific_ids)} sequences without specific IDs')

                # Generate pairs of sequences, limited to max_pairs_per_chunk
                pairs = generate_pairs(sequences_without_specific_ids, args.max_pairs_per_chunk)
                logging.info(f'Generated {len(pairs)} pairs of sequences')

                # Write pairs to the CSV file
                for pair in pairs:
                    writer.writerow(pair)

                # Check if any sequences in the output CSV correspond with sequences in the Excel file
                output_sequences = [seq.split('|')[0] for seq in sequences_without_specific_ids]
                matching_sequences = [seq for seq in output_sequences if seq in specific_ids]
                if matching_sequences:
                    logging.warning(f'Found {len(matching_sequences)} sequences in the output CSV matching sequences in the Excel file')

    logging.info('Script execution completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pairs of sequences from FASTA files")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing sequence files")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file for sequence pairs")
    parser.add_argument("--excel_file", type=str, required=True, help="Excel file containing specific IDs")
    parser.add_argument("--max_pairs_per_chunk", type=int, default=500, help="Maximum number of sequence pairs per chunk")
    args = parser.parse_args()

    main(args)

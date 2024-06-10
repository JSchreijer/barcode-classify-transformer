import os
import csv
import random
import pandas as pd
from Bio import SeqIO
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
            sequences.append(record)
    return sequences

# Function to generate pairs of sequences
def generate_pairs(items):
    pairs = []
    random.shuffle(items)
    for i in range(0, len(items) - 1, 2):
        pairs.append((items[i], items[i + 1]))
    return pairs

def main(input_folder, output_file, excel_file, remaining_output_file, max_pairs_percentage):
    # Read the specific IDs from the Excel file
    df = pd.read_excel(excel_file)
    specific_ids = set(df['ID'].tolist())

    # Lists to store sequences
    all_sequences = []
    specific_sequences = []

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            logging.info(f'Reading sequences from file: {file_path}')

            # Read sequences from the current file using Biopython
            sequences = read_fasta_sequences(file_path)

            # Separate sequences into those with and without specific IDs
            for seq_record in sequences:
                if seq_record.id.split('|')[0] in specific_ids:
                    specific_sequences.append(seq_record)
                else:
                    all_sequences.append(seq_record)

    # Calculate the number of sequences to select (80% of total)
    num_sequences_to_select = int(len(all_sequences) * max_pairs_percentage)

    # Select 80% of the total sequences
    selected_sequences = random.sample(all_sequences, num_sequences_to_select)

    # Add the remaining 20% of sequences to the specific_sequences list
    remaining_sequences = specific_sequences + [seq for seq in all_sequences if seq not in selected_sequences]

    # Open the output CSV file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Generate pairs within the selected sequences
        pairs = generate_pairs(selected_sequences)
        logging.info(f'Generated {len(pairs)} pairs of sequences')

        # Write pairs to the CSV file
        for pair in pairs:
            writer.writerow([pair[0].seq, pair[1].seq])

    # Write the remaining sequences to a separate file with the sequence and bin_id
    with open(remaining_output_file, 'w', newline='') as remaining_file:
        writer = csv.writer(remaining_file)
        writer.writerow(["seq", "bin_id"])  # Write header
        for seq_record in remaining_sequences:
            writer.writerow([seq_record.seq, seq_record.id.split('|')[0]])

    logging.info('Script execution completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sequences and generate output files.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing sequence files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for aligned sequences.")
    parser.add_argument("--excel_file", type=str, required=True, help="Excel file containing specific IDs.")
    parser.add_argument("--remaining_output_file", type=str, required=True, help="Output file for remaining sequences.")
    parser.add_argument("--max_pairs_percentage", type=float, default=0.8, help="Percentage of sequences to select for pairs (default: 0.8).")

    args = parser.parse_args()
    main(args.input_folder, args.output_file, args.excel_file, args.remaining_output_file, args.max_pairs_percentage)

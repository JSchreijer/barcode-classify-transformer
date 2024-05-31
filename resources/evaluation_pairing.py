import os
import csv
import random
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
def generate_pairs(items):
    pairs = []
    random.shuffle(items)
    for i in range(0, len(items) - 1, 2):
        pairs.append((items[i], items[i + 1]))
    return pairs

def main(input_folder, output_file, excel_file, max_pairs_percentage):
    # Read the specific IDs from the Excel file
    df = pd.read_excel(excel_file)
    specific_ids = set(df['ID'].tolist())

    # List to store all sequences from all files
    all_sequences = []
    sequences_with_specific_ids = []

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            logging.info(f'Reading sequences from file: {file_path}')

            # Read sequences from the current file using Biopython
            sequences = read_fasta_sequences(file_path)

            # Separate sequences with specific IDs
            for seq in sequences:
                if seq.split('|')[0] in specific_ids:
                    sequences_with_specific_ids.append(seq)
                else:
                    all_sequences.append(seq)

    # Calculate the number of sequences to select (80% of total)
    num_sequences_to_select = int(len(all_sequences) * max_pairs_percentage)

    # Select 80% of the total sequences
    selected_sequences = random.sample(all_sequences, num_sequences_to_select)

    # Get the remaining 20% of sequences
    remaining_sequences = [seq for seq in all_sequences if seq not in selected_sequences]

    # Combine the remaining 20% with sequences that have specific IDs
    combined_sequences = remaining_sequences + sequences_with_specific_ids

    # Open the output CSV file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Generate pairs within the combined sequences
        pairs = generate_pairs(combined_sequences)
        logging.info(f'Generated {len(pairs)} pairs of sequences')

        # Write pairs to the CSV file
        for pair in pairs:
            writer.writerow(pair)

        logging.info('Script execution completed')

if __name__ == "__main__":
    # Define your data here
    input_folder = "your_input_folder_path"  # Path to the folder containing sequence files
    output_file = "your_output_file_path.csv"  # Path to the output CSV file
    excel_file = "your_excel_file_path.xlsx"  # Path to the Excel file containing specific IDs
    max_pairs_percentage = 0.8  # Maximum percentage of pairs to generate (default: 0.8)

    main(input_folder, output_file, excel_file, max_pairs_percentage)

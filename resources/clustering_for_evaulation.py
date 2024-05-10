import os
import pandas as pd
from Bio import SeqIO

def read_excel_ids(excel_file):
    """Read IDs from Excel file."""
    df = pd.read_excel(excel_file)
    ids = set(df['ID'])
    return ids

def read_sequences_from_file(file_path):
    """Read sequences from a file."""
    sequences = []
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(record)
    return sequences

def generate_matched_sequences_tsv(input_folder, excel_file, output_file, id_output_file):
    # Read IDs from Excel file
    excel_ids = read_excel_ids(excel_file)
    all_sequence_ids = set()  # To store all unique IDs from files

    with open(output_file, 'w') as out_file, open(id_output_file, 'w') as id_file:
        out_file.write("seq\tbin_id\n")  # Add headers

        # Loop through each file in the folder
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)

                # Read sequences from the file
                sequences = read_sequences_from_file(file_path)

                # Check for matching IDs and write sequences to output file
                for record in sequences:
                    sequence_id = record.id  # Get sequence ID
                    if sequence_id in excel_ids:
                        # Write all sequences from this file along with the filename
                        bin_id = os.path.splitext(file)[0]
                        for seq_record in sequences:
                            out_file.write(f"{seq_record.seq}\t{bin_id}\n")
                            all_sequence_ids.add(seq_record.id)  # Add sequence ID to set
                        break  # Stop searching for matches in this file once one is found

        # Write all unique sequence IDs to the id_output_file
        for sequence_id in all_sequence_ids:
            id_file.write(f"{sequence_id}\n")

# Example usage:
input_folder = '../data/l0.2_s3_4_1500_o1.0_a0_constr_localpair/chunks/unaligned'
excel_file = 'Lena_result_list.xlsx'
output_file = "matched_clustering_sequences.tsv"
id_output_file = "all_sequence_ids.txt"
generate_matched_sequences_tsv(input_folder, excel_file, output_file, id_output_file)

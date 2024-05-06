import os
from Bio import SeqIO

def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file."""
    sequences = []
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))
    return sequences

def generate_sequences_tsv(input_folder, output_file, file_limit):
    with open(output_file, 'w') as out_file:
        out_file.write("seq\tbin_id\n")  # Add headers
        count = 0
        total_files = sum(1 for _ in os.listdir(input_folder))
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if count >= total_files - file_limit:
                    file_path = os.path.join(root, file)
                    sequences = read_fasta_sequences(file_path)

                    # Take the name of the file (without extension) as bin_id
                    bin_id = os.path.splitext(file)[0]

                    # Write 100 sequences for each file
                    for seq in sequences:
                        out_file.write(f"{seq}\t{bin_id}\n")
                count += 1

# Example usage:
input_folder = '../data/l0.2_s3_4_1500_o1.0_a0_constr_localpair/chunks/unaligned'
output_file = "Output_clustering_last114.tsv"
file_limit = 114
generate_sequences_tsv(input_folder, output_file, file_limit)

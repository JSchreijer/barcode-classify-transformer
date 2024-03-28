import os
import csv

# Define the input folder containing sequence files
input_folder = '../data/l0.2_s3_4_1500_o1.0_a0_constr_localpair/chunks/unaligned'

# Define the output CSV file
output_file = 'output.csv'

# Open the output CSV file in write mode
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Read sequences from the current file
            with open(file_path, 'r') as file:
                sequences = []
                current_sequence = ''
                # Iterate over lines in the file
                for line in file:
                    # Check if the line is a header line starting with ">"
                    if line.startswith(">"):
                        # Append the previous sequence if it's not empty
                        if current_sequence:
                            sequences.append(current_sequence)
                            current_sequence = ''
                    else:
                        # Concatenate the lines to form the current sequence
                        current_sequence += line.strip()

                # Append the last sequence
                if current_sequence:
                    sequences.append(current_sequence)

                # Write the pairs of sequences into two columns
                for i in range(0, len(sequences), 2):
                    # Ensure that there are two sequences to pair
                    if i + 1 < len(sequences):
                        writer.writerow([sequences[i], sequences[i + 1]])
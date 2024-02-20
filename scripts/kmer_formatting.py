#Using the DNABERT model from Github and converting the data into k-mers
from Bio import SeqIO

# now trying it with my own data
with open('../data/test.fasta', 'r') as test:
    sequences_test = list(SeqIO.parse(test, 'fasta'))

with open('../data/train.fasta', 'r') as train:
    sequences_train = list(SeqIO.parse(train, 'fasta'))
# Function to convert sequences to k-mers
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """

    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    print(kmer)
    kmers = " ".join(kmer)
    return kmers
#
# Convert each sequence in the data to k-mers
k = 6  # Specify the k-mer length

#
kmers_test = [seq2kmer(str(seq.seq), k) for seq in sequences_test]

kmers_train = [seq2kmer(str(seq.seq), k) for seq in sequences_train]
# Print the result

with open("data/kmers_test.fasta", "w") as output_file:
    for kmer_test in kmers_test:
        output_file.write(kmer_test + "\n")

with open("data/kmers_train.fasta", "w") as output_file:
    for kmer_train in kmers_train:
        output_file.write(kmer_train + "\n")

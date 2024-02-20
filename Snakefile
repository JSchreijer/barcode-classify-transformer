#dsnakefile

rule test:
    input:
        'test.txt'
    script:
        'test.py'

rule split_sequences:
    input:
        backbone = 'data/backbone.fasta',
        test = 'model/Query_ID.test1.txt' # Need to figure out how to do this with each of the test text files. Perhaps make two separate rules for this?
    output:
        'model/test.fasta',
        'model/train.fasta' # reference sequences
    params:
        dependencies='requirements.txt'
    script:
        "random_sequences.py"

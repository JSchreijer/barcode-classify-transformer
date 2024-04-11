

FROM ubuntu:latest
LABEL authors="JSchreijer"

ENTRYPOINT ["top", "-b"]

# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

#install git
RUN apt-get update && apt-get install -y git

# Clone DNABERT_S repository
RUN git clone https://github.com/MAGICS-LAB/DNABERT_S

# Set the working directory to DNABERT_S
WORKDIR /app/DNABERT_S

# Install gdown and unzip data for training
RUN pip install gdown && \
    gdown https://drive.google.com/uc?id=1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp && \
    unzip dnabert-s_train.zip && \
    gdown https://drive.google.com/uc?id=1I44T2alXrtXPZrhkuca6QP3tFHxDW98c && \
    unzip dnabert-s_eval.zip

# Install conda and create environment
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /usr/local/miniconda/bin/conda init bash && \
    conda env create -f environment.yml python=3.8 && \
    echo "conda activate DNABERT_S" >> ~/.bashrc

# Activate environment and install requirements
RUN echo "source activate DNABERT_S" && \
    pip install -r requirements.txt && \
    pip uninstall triton

# Run Python commands
COPY ./pretrain /app/pretrain
WORKDIR /app/pretrain
RUN python -c "import torch; from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-S', trust_remote_code=True); model = AutoModel.from_pretrained('zhihan1996/DNABERT-S', trust_remote_code=True); dna = 'ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC'; inputs = tokenizer(dna, return_tensors = 'pt')['input_ids']; hidden_states = model(inputs)[0]; embedding_mean = torch.mean(hidden_states[0], dim=0); print(embedding_mean.shape)"

# Continue with remaining commands
RUN export PATH_TO_DATA_DICT=/../../../../data && \
    export TRAIN_FILE=aligned_sequences.csv && \
    python main.py \
        --resdir ./results/ \
        --datapath ${PATH_TO_DATA_DICT} \
        --train_dataname ${TRAIN_FILE} \
        --val_dataname val_48k.csv \
        --seed 1 \
        --logging_step 10000 \
        --logging_num 12 \
        --max_length 2000 \
        --train_batch_size 48 \
        --val_batch_size 360 \
        --lr 3e-06 \
        --lr_scale 100 \
        --epochs 3 \
        --feat_dim 128 \
        --temperature 0.05 \
        --con_method same_species \
        --mix \
        --mix_alpha 1.0 \
        --mix_layer_num -1 \
        --curriculum

RUN export MODEL_DIR=/barcode-classify-transformer/resources/DNABERT_S-main/pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs8.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best && \
    cp model_codes/* ${MODEL_DIR}

RUN export DATA_DIR=. && \
    python eval_clustering_classification.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test" && \
    python eval_clustering_classification.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"

RUN export DATA_DIR=. && \
    export MODEL_DIR=/barcode-classify-transformer/resources/DNABERT_S-main/models/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs8.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best && \
    python eval_binning.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test" && \
    python eval_binning.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"

RUN export DATA_DIR=. && \
    export MODEL_DIR=../pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs8.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best && \
    python eval_binning.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"

# barcode-classify-transformer
Repository for experiments with AI transformer model-based barcode classification.
Internship by Jitske Schreijer

In this thesis I will be working with data from https://github.com/luukromeijn/MDDB-phylogeny/tree/main 
and implementing this, using a transformer model. This transformer model will be based on DNABERT-S. 
TThe final version will be created as a dockerfile where the entire code is visible with all data download automatically downloaded. 
However, in this README, I will show the steps that were taken to create said dockerfile one by one.
```
@misc{zhou2024dnaberts,
      title={DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models}, 
      author={Zhihan Zhou and Winmin Wu and Harrison Ho and Jiayi Wang and Lizhen Shi and Ramana V Davuluri and Zhong Wang and Han Liu},
      year={2024},
      eprint={2402.08777},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}

}
```
# Environment setup
For this research,a virtual conda environment using snakemake was created. this will be done using an 64-bit Ubuntu Linux environment. 
Make sure the virtual environment is created using python3.8. Otherwise, the package transformers will not be able to download. 

An easy way to link the Ubuntu Linux environment with the snakefiles and python files, is to use the interface "JetBrains Gateway". 
Here, the terminal can be used as virtual environment and switching between the WSL terminal and python won't be necessary. 


Afterwards, a working directory is created using the mkdir function and calling it using the cd function. 
The environment is activated using conda. 

For the creation of DNABERT-S, the following packages are required: 
```
# command line
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data 

gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c # pip install gdown
unzip dnabert-s_eval.zip  # unzip the data 

#The training data that was used from the MDDB phylogeny will be added on Zenodo. 

conda create -n DNABERT_S python=3.9
conda activate DNABERT_S

pip install -r requirements.txt
pip uninstall triton # this can lead to errors in GPUs other than A100


## in python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

## embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

```
## Training the model 

First we need to create the training data for the model. For this model, we use the data from MDDB-phylogeny. More specifically, we use the recommended dataset, called [l0.2_s3_4_1500_o1.0_a0_constr_localpair](data%2Fl0.2_s3_4_1500_o1.0_a0_constr_localpair). 
From this dataset, the unaligned chunks folder is used.  
To create a dataset in the correct format, sequence_pairing.py can be used to create pairwise aligned sequences in rows with two columns and no headers. 
This script runs throught all of the unaligned folders and creates pairs from sequences within each chunk. It also skips over any sequence ID that is used in the results table of "ten Haaf, L., Verbeek, F., & Vos, R. (2022). BSc Bioinformatics Localized Information Comparison and Analysis for MycoDiversity Database."
This is done to make sure the final results can compared to one another. 
Afterwards, the model can be trained: 
```
cd pretrain 
export PATH_TO_DATA_DICT=../../
export TRAIN_FILE=aligned_sequences.csv

python main.py \
    --resdir ./results/ \
    --datapath ${PATH_TO_DATA_DICT} \
    --train_dataname ${TRAIN_FILE} \
    --val_dataname val_48k.csv \
    --seed 1 \
    --logging_step 10000 \
    --logging_num 12 \
    --max_length 20 \
    --train_batch_size 6 \
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

``` 
This is the model training based on the dataset from the MDDB-phylogeny. 
The aligned_sequences is the trainingset (see Lena's thesis) and repurposed into a csv file of two rows and no columns.

Afterwards, the necessary files are copied to the folder where the model is saved. This is a bug in Huggingface Transformers package.
Sometimes the model file such as bert_layer.py are not automatically saved to the model directory together with the model weights. So we manually do it.
```
# using the evaluate folder 
cd evaluate

#exporting model the best model 
export MODEL_DIR=../pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs6.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best

cp model_codes/* ${MODEL_DIR}
```

Afterwards, we opted for a KNN evaluation instead of the traditional evaluation and used the code of the original to create the KNN evaluation on the created embeddings. 
I got this error: 

  raise EnvironmentError(
OSError: ../pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs6.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best does not appear to have a file named zhihan1996/DNABERT-2-117M--configuration_bert.py. Checkout 'https://huggingface.co/../pretrain/results/epoch3.aligned_sequences.csv.lr3e-06.lrscale100.bs6.maxlength20.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/best/None' for available files.

I solved this by changing the settings of config.json: "attention_probs_dropout_prob": 0.0,
"auto_map": {
    "AutoConfig": "configuration_bert.BertConfig",
    "AutoModel": "bert_layers.BertModel",
    "AutoModelForMaskedLM": "bert_layers.BertForMaskedLM",
    "AutoModelForSequenceClassification": "bert_layers.BertForSequenceClassification"
},

This was different for the other model that DID work. 

Now for the evaluation I created two new files called KNN_evaluation.py and linear_probing_evaluation.py

These two work with argparse:
```

KNN_evaluation.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"
Linear_probing_evaluation.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"
```
These two evaluation scripts create a kNN and linear probing algorithm respectively, using the "evaluation_sequences.tsv" file that was created for that purpose. 
They each give a value for the accuracy of the trained model and create a t-SNE for visualising purposes.


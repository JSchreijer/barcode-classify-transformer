# barcode-classify-transformer
Repository for experiments with AI transformer model-based barcode classification.
Internship by Jitske Schreijer

In this thesis I will be working with data from https://github.com/luukromeijn/MDDB-phylogeny/tree/main 
and implementing this, using a transformer model. This transformer model will be based on DNABERT-S
```
 @misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```
# Environment setup
For this research,a virtual conda environment using snakemake was created. this will be done using an 64-bit Ubuntu Linux environment. 
Make sure the virtual environment is created using python3.8. Otherwise, the package transformers will not be able to download. 

An easy way to link the Ubuntu Linux environment with the snakefiles and python files, is to use the interface "JetBrains Gateway". 
Here, the terminal can be used as virtual environment and switching between the WSL terminal and python won't be necessary. 

```
Afterwards, a working directory is created using the mkdir function and calling it using the cd function. 
The environment is activated using conda. 

For the creation of DNABERT-S, the following packages are required: 
```
# command line
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data 
 
pip install -r requirements.txt

# in python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768
```


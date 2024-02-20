# barcode-classify-transformer
Repository for experiments with AI transformer model-based barcode classification.
Internship by Jitske Schreijer

In this thesis I will be working with data from https://github.com/luukromeijn/MDDB-phylogeny/tree/main 
and implementing this, using a transformer model. This transformer model will be based on DNABERT
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
For this research,a virtual environment using snakemake was created. this will me done using the  
vagrant virtual machine, creating an 64-bit Ubuntu Linux environment.

the new virtual environment will be created and activated using
```
> vagrant init hashicorp/precise64
> vagrant up
```

Next, Mambaforge has to be installed, via the Vagrant Linux VM. 
```
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -o Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh

```
Afterwards, a working directory is created using the mkdir function and calling it using the cd function. 
The environment is activated using conda. 

For the creation of DNABERT, the following packages are required: 
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```
Afterwards, the data has to be conerted into kmer formatting using the seq2kmer function from scripts/kmer_formatting.py


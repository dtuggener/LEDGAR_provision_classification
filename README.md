# Code for the paper "LEDGAR: A Large-Scale Multilabel Corpus for Text Classification of Legal Provisions in Contracts"

## Requirements:
* Python 3.7+
* to install requirements: `pip install -r requirements.txt`

## Data:

* The full corpus as a zipped jsonl file
 is located [here](https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A).
 
 * for the MLP (+Attention) classification experiments you will also need 
 pretrained MUSE embeddings from [here](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec).
 
 * prepare word embeddings: `python convert_embedding_txt.py /path/to/wiki.multi.en.vec`
 this will create `wiki.multi.en.vec_data.npy` and `wiki.multi.en.vec_vocab.json` in the
 same folder.

## Usage:

* creating the different sub-corpora:
 `python corpus_analysis_and_sampling.py /path/to/LEDGAR_2016-2019_clean.jsonl`

* to run the classification baselines, navigate do the classification sub folder: 
`cd classification`
* Logistic Regression: `python classification_baselines.py /path/to/sub-corpus.jsonl`
* MLP: `python mlp_classifier.py /path/to/sub-corpus.jsonl /path/to/wiki.multi.en.vec_data.npy /path/to/wiki.multi.en.vec_vocab.json`
* MLP + Attention: `python mlp_classifier_attention.py /path/to/sub-corpus.jsonl /path/to/wiki.multi.en.vec_data.npy /path/to/wiki.multi.en.vec_vocab.json`
* DistilBert: `python distilbert_baseline.py --data /path/to/sub-corpus.jsonl --mode train`
for more detailed instructions consult [this readme](./classification/distilbert.md).

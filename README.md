# Code for the paper "LEDGAR: A Large-Scale Multilabel Corpus for Text Classification of Legal Provisions in Contracts"
Link to the paper: https://www.aclweb.org/anthology/2020.lrec-1.155.pdf
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

## Citation:
Thanks for citing our paper should you use the corpus!
```
@inproceedings{tuggener2020ledgar,
  title={LEDGAR: a large-scale multi-label corpus for text classification of legal provisions in contracts},
  author={Tuggener, Don and von D{\"a}niken, Pius and Peetz, Thomas and Cieliebak, Mark},
  booktitle={12th Language Resources and Evaluation Conference (LREC) 2020},
  pages={1228--1234},
  year={2020},
  organization={European Language Resources Association}
}
```

# Code for the paper "LEDGAR: A Large-Scale Multilabel Corpus for Text Classification of Legal Provisions in Contracts"

## Requirements:
* Python 3.7+
* to install requirements: `pip install -r requirements.txt`

## Data:

* The full corpus as a zipped jsonl file
 is located at: [https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A]

## Usage:

* creating the different sub-corpora:
 `python corpus_analysis_and_sampling.py /path/to/LEDGAR_2016-2019_clean.jsonl`

* to run the classification baselines, navigate do the classification sub folder: 
`cd classification`
* Logistic Regression: `python classification_baselines.py /path/to/sub-corpus.jsonl`
* MLP: `` _TODO_
* MLP + Attention: `` _TODO_
* DistilBert: `python distilbert_baseline.py --data /path/to/sub-corpus.jsonl --mode train`
for more detailed instructions consult: [classification/distilbert.md]

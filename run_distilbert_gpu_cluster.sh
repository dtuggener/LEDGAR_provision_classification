#!/bin/bash
cd /cluster/home/vode/sec_edgar_provision_classification
python3 -m pip install -r requirements.txt --user
python3 -m distilbert_baseline --data ./data/sec_corpus_2016-2019_clean_NDA_PTs.jsonl --mode dev --epoch 1 --batch_size 8

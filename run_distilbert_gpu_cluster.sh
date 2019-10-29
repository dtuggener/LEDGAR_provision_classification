#!/bin/bash
cd /cluster/home/tuge/sec_edgar_provision_classification
python3 -m pip install -r requirements.txt --user
python3 -m distilbert_baseline --data ./data/sec_corpus_2016-2019_clean_freq100.jsonl --mode test --model_path distill_bert_freq100.pkl --epoch 1 --batch_size 128

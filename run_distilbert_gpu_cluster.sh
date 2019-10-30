#!/bin/bash
cd /cluster/home/tuge/sec_edgar_provision_classification
python3 -m pip install -r requirements.txt --user
python3 -m distilbert_baseline --data ./data/sec_corpus_2016-2019_clean_projected_real_roots.jsonl --mode train --model_path distill_bert_leaves.pkl --epoch 3 --batch_size 8

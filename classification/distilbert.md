main usage: `python -m distilbert_baseline --data /path/to/your.jsonl --mode train_or_test`

Arguments:
* `-h` or `--help`: show a help message and exit
* `--data DATA`: Path to .jsonl file containing dataset 
* `--mode MODE`: which testing mode: 'train' or 'test'
* `--use_class_weights USE_CLASS_WEIGHTS`: use balanced class weights for training, default True
* `--seed SEED`: seed for random number generation, default 0xDEADBEEF
* `--max_seq_len MAX_SEQ_LEN`: maximum sequence length in transformer, default 128
* `--batch_size BATCH_SIZE`: training batch size, default 8
* `--epochs EPOCHS`: number of epochs of training, default 1
* `--weight_decay WEIGHT_DECAY`: AdamW weight decay, default 0.0
* `--learning_rate LEARNING_RATE`: AdamW learning rate, default 5e-5
* `--adam_epsilon ADAM_EPSILON`: AdamW epsilon, default 1e-8
* `--warmup_steps WARMUP_STEPS`: Warmup steps for learning rate schedule, default 0
* `--max_grad_norm MAX_GRAD_NORM`: max norm for gradient clipping, default 1.0

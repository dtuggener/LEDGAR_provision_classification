
# How to run a distilbert experiment

## setup
for more in depth info consult the
[cloudlab info page](https://info.cloudlab.zhaw.ch/pages/gpu/getting_started.html)

* connect to cluster: `ssh user@gpulogin.cloudlab.zhaw.ch`
* pull latest pytorch docker image: `singularity pull docker://pytorch/pytorch:latest`
    this will create a file `pytorch-latest.simg`
* clone *sec_edgar_provision_classification* repository
* change the `cd` command in *run_distilbert_gpu_cluster.sh* to navigate to where your repository is located


## run an experiment

* modify *run_distilbert_gpu_cluster.sh* to run *distilbert_baseline.py* with your
preferred command line arguments
* run `srun --pty --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 singularity exec pytorch-latest.simg /path/to/run_distilbert_gpu_cluster.sh`

Note that the `srun` command is assumed to be run in your home directory.

## command line arguments

main usage: `python -m distilbert_baseline --data /path/to/your.jsonl --mode dev_or_test`
Comment: I found fiddling with `--batch_size` to be the most fruitful

Arguments:
* `-h` or `--help`: show a help message and exit
* `--data DATA`: Path to .jsonl file containing dataset. *Note* that if you follow the
steps above the path will be relative to the repository directory, when in doubt, use
absolute path
* `--mode MODE`: which testing mode: 'dev' or 'test'
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

import random

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from pytorch_transformers import (
    WEIGHTS_NAME,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from sklearn.metrics import classification_report

from tqdm import tqdm, trange
import numpy as np

from distilbert_data_utils import DonData, convert_examples_to_features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(train_dataset, model):
    # TODO: magic numbers, defaults in run_glue.py
    batch_size = 32
    n_epochs = 3
    weight_decay = 0.0
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    seed = 0xdeafbeef
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_grad_norm = 1.0

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
    )

    no_decay = {'bias', 'LayerNorm.weight'}
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon,
    )
    scheduler = WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        t_total=len(train_dataloader) // n_epochs,
    )

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iter = trange(n_epochs, desc='Epoch')
    set_seed(seed=seed)
    for _ in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iter):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                #'token_type_ids': batch[2],  # probably used for distilbert
                'labels': batch[3],
            }

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

    return global_step, tr_loss / global_step


def evaluate(eval_dataset, model):
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size
    )

    preds = None
    out_label_ids = None
    for batch in tqdm(eval_loader, desc="Evaluation"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                #'token_type_ids': batch[2],
                'labels': batch[3]
            }
            outputs = model(**inputs)
            logits = outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0,
                )

    preds = np.argmax(preds, axis=1)

    return {
        'pred_ids': preds,
        'true_ids': out_label_ids,
    }


def main():
    max_seq_length = 128

    don_data = DonData(path='./data/sec_corpus_2016-2019_clean_proto.jsonl')

    model_name = 'distilbert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DistilBertConfig.from_pretrained(model_name, num_labels=len(don_data.all_lbls))
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    model.to(device)

    # training
    print('construct training data tensor')
    train_data = convert_examples_to_features(
        examples=don_data.train(),
        label_list=don_data.all_lbls,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('start training')
    train(train_dataset=train_data, model=model)

    # eval
    print('construct test data tensor')
    eval_data = convert_examples_to_features(
        examples=don_data.test(),
        label_list=don_data.all_lbls,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('predict test set')
    prediction_data = evaluate(eval_dataset=eval_data, model=model)

    print(classification_report(
        y_true=prediction_data['true_ids'],
        y_pred=prediction_data['pred_ids'],
        labels=list(range(len(don_data.all_lbls))),
        target_names=don_data.all_lbls,
    ))


if __name__ == '__main__':
    main()

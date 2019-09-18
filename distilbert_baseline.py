
import random

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from pytorch_transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_distilbert import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)

from sklearn.metrics import f1_score, classification_report

from tqdm import tqdm, trange
import numpy as np

from distilbert_data_utils import DonData, convert_examples_to_features
from utils import evaluate_multilabels


class DistilBertForMultilabelSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertForMultilabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            head_mask=None,
            labels=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss(
                    reduction='mean',
                )
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs


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

    preds = np.tanh(preds)

    return {
        'pred': preds,
        'truth': out_label_ids,
    }


def tune_threshs(probas, truth):
    res = np.zeros(probas.shape[1])

    for i in range(probas.shape[1]):
        thresh = max(
            np.linspace(-1.0, 1.0, num=100),  # we use tanh instead of sigmoid so it's not technicallly probas
            key=lambda t: f1_score(y_true=truth[:, i], y_pred=(probas[:, i] > t))
        )
        res[i] = thresh

    return res


def apply_threshs(probas, threshs):
    res = np.zeros(probas.shape)

    for i in range(probas.shape[1]):
        res[:, i] = probas[:, i] > threshs[i]

    return res


def multihot_to_label_lists(label_array, label_map):
    label_id_to_label = {
        v: k
        for k, v in label_map.items()
    }
    res = []
    for i in range(label_array.shape[0]):
        lbl_set = []
        for j in range(label_array.shape[1]):
            if label_array[i, j] > 0:
                lbl_set.append(label_id_to_label[j])
        res.append(lbl_set)
    return res


def main():
    max_seq_length = 128

    don_data = DonData(path='./data/sec_corpus_2016-2019_clean_proto.jsonl')

    model_name = 'distilbert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DistilBertConfig.from_pretrained(model_name, num_labels=len(don_data.all_lbls))
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = DistilBertForMultilabelSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    model.to(device)

    # training
    print('construct training data tensor')
    train_data = convert_examples_to_features(
        examples=don_data.train(),
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('start training')
    train(train_dataset=train_data, model=model)

    # eval
    print('construct test data tensor')
    eval_data = convert_examples_to_features(
        examples=don_data.test(),
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('predict test set')
    prediction_data = evaluate(eval_dataset=eval_data, model=model)

    # tune thresholds
    print('tuning clf thresholds')
    threshs = tune_threshs(
        probas=prediction_data['pred'],
        truth=prediction_data['truth'],
    )
    predicted_mat = apply_threshs(
        probas=prediction_data['pred'],
        threshs=threshs,
    )

    print("Result:")
    evaluate_multilabels(
        y=multihot_to_label_lists(prediction_data['truth'], don_data.label_map),
        y_preds=multihot_to_label_lists(predicted_mat, don_data.label_map),
        do_print=True,
    )


if __name__ == '__main__':
    main()

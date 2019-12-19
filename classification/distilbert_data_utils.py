
import itertools

import torch
from torch.utils.data import TensorDataset

from utils import split_corpus

import numpy as np


def multihot(labels, label_map):
    res = np.zeros(len(label_map))
    for lbl in labels:
        res[label_map[lbl]] = 1.0
    return res


class DonData(object):

    def __init__(self, path):
        self.don_data = split_corpus(path)
        self.all_lbls = list(sorted({
            label
            for lbls in itertools.chain(
                self.don_data.y_train,
                self.don_data.y_test,
                self.don_data.y_dev if self.don_data.y_dev is not None else []
            )
            for label in lbls
        }))
        self.label_map = {
            label: i
            for i, label in enumerate(self.all_lbls)
        }

        total = 0
        self.class_weights = np.zeros(len(self.label_map), dtype=np.float32)
        for sample in self.train():
            self.class_weights += sample['label']
            total += 1
        self.class_weights = total / (len(self.label_map) * self.class_weights)

    def train(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_train, self.don_data.y_train)]

    def test(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_test, self.don_data.y_test)]

    def dev(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_dev, self.don_data.y_dev)]


class ListData(object):

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        assert len(self.xs) == len(self.ys)

    def label_list(self):
        return list(sorted(set(self.ys)))

    def examples(self):
        return [{
            'txt': x,
            'label': y,
        } for x, y in zip(self.xs, self.ys)]


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_segment_id=0,
        sequence_segment_id=0,
        mask_padding_with_zero=True,
):
    # / ! \ copy-pasted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py
    # should work with bert and distilbert, will return dataset directly
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    all_input_ids = []
    all_input_masks = []
    all_segment_ids = []
    all_label_ids = []
    for ex_ix, example in enumerate(examples):

        tokens = tokenizer.tokenize(example['txt'])

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        segment_ids = [sequence_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        pad_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * pad_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * pad_length) + input_mask
            segment_ids = ([pad_token_segment_id] * pad_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * pad_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * pad_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * pad_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        all_input_ids.append(input_ids)
        all_input_masks.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(example['label'])

    input_id_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(all_input_masks, dtype=torch.long)
    segment_id_tensor = torch.tensor(all_segment_ids, dtype=torch.long)
    label_id_tensor = torch.tensor(all_label_ids, dtype=torch.float)

    return TensorDataset(
        input_id_tensor,
        input_mask_tensor,
        segment_id_tensor,
        label_id_tensor,
    )

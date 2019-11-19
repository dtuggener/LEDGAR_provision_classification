import json
import random; random.seed(42)
from collections import Counter
from typing import List
from corpus_analysis_and_sampling import write_jsonl


def shuffle_data(x, y, d):
    """ Randomly shuffle co-indexed lists, pertaining the alignment of the lists"""
    xyd = list(zip(x, y, d))
    random.shuffle(xyd)
    x, y, d = zip(*xyd)
    x, y, d = list(x), list(y), list(d)
    return x, y, d


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean_freq100.jsonl'

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data from', corpus_file)
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    labels = [l for labels in y for l in labels]
    label_counts = Counter(labels)

    min_freq = 100
    target_freq = 100

    x, y, doc_ids = shuffle_data(x, y, doc_ids)

    seen_labels = Counter()
    sampled_x, sampled_y, sampled_docids = [], [], []
    for text, labels, doc_id in zip(x, y, doc_ids):
        for label in labels:
            if label_counts[label] >= min_freq and seen_labels[label] < target_freq:
                sample_labels = [l for l in labels if label_counts[l] >= min_freq]
                for l in sample_labels:
                    seen_labels[l] += 1
                sampled_x.append(text)
                sampled_y.append(sample_labels)
                sampled_docids.append((doc_id))
                break

    sampled_labels = [l for labels in sampled_y for l in labels]
    sampled_label_counts = Counter(sampled_labels)

    write_jsonl(corpus_file.replace('.jsonl', '_subsampled.jsonl'),
        sampled_x,
        sampled_y,
        sampled_docids)



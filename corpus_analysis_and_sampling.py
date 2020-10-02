import json
import numpy
import matplotlib.pyplot as plt
import re
from typing import List
from collections import defaultdict, Counter


def label_stats(x, y, doc_ids, n=10):
    labels = [l for labels in y for l in labels]
    label_counts = Counter(labels)
    print(len(label_counts), 'labels,', len(x), 'provisions', len(set(doc_ids)), 'contracts')
    for label, cnt in label_counts.most_common(n):
        print(label, cnt)
    ml = [(text, labels) for text, labels in zip(x, y) if len(labels) > 1]
    print('{} provisions with multilabels ({}%)'.format(len(ml), round(100*len(ml)/len(y), 2)))


def sample_frequent_labels(x, y, doc_ids, min_freq=None, max_freq=None, n_labels=None):
    label_counts = Counter([l for labels in y for l in labels])
    selected_labels = label_counts.copy()

    if min_freq:
        selected_labels = Counter({l: c for l, c in selected_labels.items() if c >= min_freq})
    if max_freq:
        selected_labels = Counter({l: c for l, c in selected_labels.items() if c <= max_freq})
    if n_labels:
        selected_labels = Counter({l: c for (l, c) in label_counts.most_common(n_labels)})

    x_small, y_small, doc_ids_small = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        sel_labels = [l for l in labels if l in selected_labels]
        if sel_labels:
            x_small.append(provision)
            y_small.append(sel_labels)
            doc_ids_small.append(doc_id)
    return x_small, y_small, doc_ids_small


def sample_common_labels(x, y, doc_ids, n_labels=20):
    labels2docs = defaultdict(set)
    for provision, labels, doc_id in zip(x, y, doc_ids):
        for label in labels:
            labels2docs[label].add(doc_id)
    labels2docs_counts = Counter({l: len(ds) for l, ds in labels2docs.items()})
    selected_labels = [l for (l, _) in labels2docs_counts.most_common(n_labels)]

    filt_x, filt_y, filt_doc_ids = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        filt_labels = []
        for label in labels:
            if label in selected_labels:
                filt_labels.append(label)
        if filt_labels:
            filt_x.append(provision)
            filt_y.append(filt_labels)
            filt_doc_ids.append(doc_id)
    return filt_x, filt_y, filt_doc_ids


def avg_provision_count(y, doc_ids):
    doc2labels = defaultdict(list)
    for labels, doc_id in zip(y, doc_ids):
        doc2labels[doc_id].append(labels)
    doc2labels_counts = Counter({doc_id: len(labels) for doc_id, labels in doc2labels.items()})
    avg_prov_count = int(numpy.mean(list(doc2labels_counts.values())))
    return avg_prov_count


def write_jsonl(out_file: str, x_small, y_small, doc_ids_small):
    print('Writing output')
    with open(out_file, 'w', encoding='utf8') as f:
        for provision, labels, doc_id in zip(x_small, y_small, doc_ids_small):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')


def create_subcorpora(x, y, doc_ids):
    print('Sampling most common provisions')
    avg_prov_cnt = avg_provision_count(y, doc_ids)  # Average no. of provisions per contract
    x_small, y_small, doc_ids_small = sample_common_labels(x, y, doc_ids, n_labels=avg_prov_cnt)
    label_stats(x_small, y_small, doc_ids_small)
    out_file = corpus_file.replace('.jsonl', '_proto.jsonl')
    write_jsonl(out_file, x_small, y_small, doc_ids_small)

    print('Sampling provisions with frequency >= 100')
    x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, min_freq=100)
    label_stats(x_small, y_small, doc_ids_small)
    out_file = corpus_file.replace('.jsonl', '_freq100.jsonl')
    write_jsonl(out_file, x_small, y_small, doc_ids_small)


def incremental_label_stats(x, y, doc_ids):
    for i in [0, 10, 50, 100, 500, 1000, 5000, 10000]:
        print(i)
        x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, min_freq=i)
        label_stats(x_small, y_small, doc_ids_small, n=0)


def plot_label_name_vs_freq(y):
    label_list = [l for labels in y for l in labels]
    label_counts_counter = Counter(label_list)
    name_lengths = []
    label_counts = []
    for label, cnt in label_counts_counter.most_common():
        label_counts.append(cnt)
        name_lengths.append(label.count(' ') + 1)
    plt.scatter(label_counts, name_lengths, marker='+', c='black')
    plt.xlabel('Label frequency')
    plt.ylabel('Label name token count')
    plt.savefig('label_name_length_vs_freq.pdf')


if __name__ == '__main__':

    import sys
    corpus_file = sys.argv[1]

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data from', corpus_file)
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    vocab = set()
    token_counts, provisions_per_doc = [], []
    curr_doc, provision_counts = '', 0
    for sample, doc_id in zip(x, doc_ids):
        if not doc_id == curr_doc:
            curr_doc = doc_id
            provisions_per_doc.append(provision_counts)
            provision_counts = 0
        provision_counts += 1
        tokens = re.findall('\w+', sample.lower())
        token_counts.append((len(tokens)))
        vocab.update(tokens)
    print('Total tokens', sum(token_counts))
    print('Mean token count', numpy.mean(token_counts))
    print('Standard deviation', numpy.std(token_counts))
    print('Vocabulary size', len(vocab))
    print('Mean provision count per doc', numpy.mean(provisions_per_doc))
    print('Standard deviation', numpy.std(provisions_per_doc))

    label_stats(x, y, doc_ids, n=0)

    plot_label_name_vs_freq(y)

    incremental_label_stats(x, y, doc_ids)

    create_subcorpora(x, y, doc_ids)



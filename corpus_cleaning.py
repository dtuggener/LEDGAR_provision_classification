import numpy
import re
import matplotlib.pyplot as plt
from typing import List, Set
from collections import defaultdict, Counter
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords


def unique_data(x, y, doc_ids):
    """Remove duplicate provisions"""
    print('Removing duplicate provisions')
    seen = set([])
    uniq_x, uniq_y, uniq_doc_ids = [], [], []
    for x_, y_, doc_id in zip(x, y, doc_ids):
        if not x_ in seen:
            uniq_x.append(x_)
            uniq_y.append(y_)
            uniq_doc_ids.append(doc_id)
            seen.add(x_)
    print('Reduced provisions from', len(x), 'to', len(uniq_x))
    return uniq_x, uniq_y, uniq_doc_ids


def split_conjuncted_labels(y: List[List[str]]) -> List[List[str]]:
    # Determine splitable labels
    label_set = {l for labels in y for l in labels}
    split_regex = re.compile(' and | & |, ')
    splitable_labels = set([])
    stop_words = set(stopwords.words('english'))
    for label in label_set:
        labels = split_regex.split(label)
        if any(l for l in labels if len(l.split(' ')[-1]) > 1 and l.split(' ')[-1] in stop_words):
            continue
        if len(labels) > 1 and any(l for l in labels if l in label_set):
            splitable_labels.add(label)

    print(len(splitable_labels), 'splitable labels')

    split_y: List[List[str]] = []
    for labels in y:
        split_labels = set([])
        for label in labels:
            if label in splitable_labels:
                split_label = split_regex.split(label)
                for l in split_label:
                    split_labels.add(l)
            else:
                split_labels.add(label)
        split_y.append(list(split_labels))

    return split_y


def merge_plural_label_names(y: List[List[str]]) -> List[List[str]]:
    mergeable_labels = set([])
    label_set = {l for labels in y for l in labels}
    for label in label_set:
        if label[-1] == 's':
            if label[:-1] in label_set:
                mergeable_labels.add(label[:-1])

    print(len(mergeable_labels), 'mergeable labels')

    merged_y: List[List[str]] = []
    for labels in y:
        merged_labels = set([])
        for label in labels:
            if label in mergeable_labels:
                merged_labels.add(label + 's')
            else:
                merged_labels.add(label)
        merged_y.append(list(merged_labels))

    return merged_y


def get_outlier_labels(y, doc_ids, do_plot=False) -> Set[str]:
    """Detect label outliers based on label frequency and label document frequency"""
    print('Detecting outlier labels...')
    label_counts = Counter()
    labels2docs = defaultdict(list)
    for labels, doc_id in zip(y, doc_ids):
        for label in labels:
            label_counts[label] += 1
            labels2docs[label].append(doc_id)

    label_set = list(label_counts.keys())
    label_counts_values, label_doc_counts_values = [], []
    for label in label_set:
        label_counts_values.append(label_counts[label])
        label_doc_counts_values.append(len(set(labels2docs[label])))

    label_counts_values = numpy.array(label_counts_values)
    label_doc_counts_values = numpy.array(label_doc_counts_values)

    lr = LinearRegression().fit(label_counts_values.reshape(-1, 1),
                                label_doc_counts_values.reshape(-1, 1))
    pred = lr.predict(label_counts_values.reshape(-1, 1)).flatten()

    dists = []
    max_dc = numpy.max(label_doc_counts_values)
    for label, dc, pdc in zip(label_set, label_doc_counts_values, pred):
        dist = (pdc - dc) / (dc/max_dc)
        dists.append((dist, label))
    std_dist = numpy.std([d[0] for d in dists])
    outlier_ixs, outliers = [], []
    for i, (dist, label) in enumerate(dists):
        if dist > std_dist:
            outliers.append(label)
            outlier_ixs.append(i)

    if do_plot:
        plt.scatter(label_counts_values, label_doc_counts_values, marker='+')
        plt.plot(label_counts_values, pred, c='green')
        plt.xlabel('Label frequency')
        plt.ylabel('Label document frequency')
        outlier_counts = numpy.take(label_counts_values, outlier_ixs)
        outlier_doc_counts = numpy.take(label_doc_counts_values, outlier_ixs)
        plt.plot(outlier_counts, outlier_doc_counts, c='red', marker='o', fillstyle='none', linestyle='')
        plt.savefig('label_outliers.pdf')

    print(len(outliers), 'outliers found')
    return set(outliers)


def identify_lowfreq_labels(x, y, doc_ids, min_freq: int = None, min_doc_freq: int = None):
    """Remove labels with low frequency"""
    print('Identifying low-frequency labels')
    if min_freq:
        label_counts = Counter(l for labels in y for l in labels)
        lowfreq_labels = {l for l, c in label_counts.items() if c < min_freq}

    elif min_doc_freq:
        labels2docs = defaultdict(set)
        for labels, doc_id in zip(y, doc_ids):
            for label in labels:
                labels2docs[label].add(doc_id)
        lowfreq_labels = {l for l, ds in labels2docs.items() if len(ds) < min_doc_freq}

    else:
        lowfreq_labels = set()

    print(len(lowfreq_labels), 'labels found')
    return lowfreq_labels


def remove_labels(x, y, doc_ids, drop_labels: Set[str]):
    filt_x, filt_y, filt_doc_ids = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        filt_labels = []
        for label in labels:
            if not label in drop_labels:
                filt_labels.append(label)
        if filt_labels:
            filt_x.append(provision)
            filt_y.append(filt_labels)
            filt_doc_ids.append(doc_id)
    return filt_x, filt_y, filt_doc_ids


if __name__ == '__main__':

    import json

    corpus_file = 'sec_corpus_2016-2019.jsonl'

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data from', corpus_file)
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')
    label_counts = Counter([l for labels in y for l in labels])

    x, y, doc_ids = unique_data(x, y, doc_ids)
    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')

    y = split_conjuncted_labels(y)
    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')

    y = merge_plural_label_names(y)
    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')

    lowfreq_labels = identify_lowfreq_labels(x, y, doc_ids, min_doc_freq=5)
    x, y, doc_ids = remove_labels(x, y, doc_ids, drop_labels=lowfreq_labels)
    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')

    outlier_labels = get_outlier_labels(y, doc_ids, do_plot=True)
    x, y, doc_ids = remove_labels(x, y, doc_ids, drop_labels=outlier_labels)
    label_set = set(l for labels in y for l in labels)
    ml = 100 * len([l for l in y if len(l) > 1]) / len(y)
    print(len(label_set), 'labels', round(ml, 2), '% multilabels', len(x), 'provisions', len(set(doc_ids)), 'contracts')

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_clean.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(x, y, doc_ids):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')

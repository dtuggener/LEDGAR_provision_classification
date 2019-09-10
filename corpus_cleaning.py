import numpy
import re
import matplotlib.pyplot as plt
from typing import List, Set
from collections import defaultdict, Counter
from sklearn.linear_model import LinearRegression


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
    for label in label_set:
        labels = split_regex.split(label)
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
    dists.sort(reverse=True)
    std_dist = numpy.std([d[0] for d in dists])
    outliers = [label for (dist, label) in dists if dist > std_dist]

    if do_plot:
        plt.scatter(label_counts_values, label_doc_counts_values, marker='+')
        plt.plot(label_counts_values, pred, c='green')
        plt.xlabel('Label frequency')
        plt.ylabel('Label document frequency')
        for label in outliers:
            ix = label_set.index(label)
            plt.plot(label_counts_values[ix], label_doc_counts_values[ix], c='red', marker='o', fillstyle='none')
        plt.savefig('label_outliers.pdf')

    """
    from sklearn.svm import SVR
    svr= SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(label_counts_values.reshape(-1, 1), label_doc_counts_values)
    svr_pred = svr.predict(label_counts_values.reshape(-1, 1))

    svr_dists = []
    for label, dc, pdc in zip(label_set, label_doc_counts_values, svr_pred):
        dist = (pdc - dc) / (dc/max_dc)
        svr_dists.append((dist, label))
    svr_dists.sort(reverse=True)
    std_dist_svr = numpy.std([d[0] for d in dists])
    svr_outliers = [(dist, label) for (dist, label) in svr_dists if dist > std_dist_svr]
    plt.scatter(label_counts_values, label_doc_counts_values, marker='+')
    plt.scatter(label_counts_values, svr_pred, c='red', marker='x')
    for _, label in svr_outliers:
        ix = label_set.index(label)
        plt.plot(label_counts_values[ix], label_doc_counts_values[ix], c='red', marker='o', fillstyle='none')
    pdb.set_trace()
    """
    print(len(outliers), 'outliers found')
    return set(outliers)


def remove_lowfreq_labels(x, y, doc_ids, min_freq: int = 2):
    """Remove labels with low frequency"""
    print('Removing low-frequency labels')
    label_counts = Counter(l for labels in y for l in labels)
    filt_x, filt_y, filt_doc_ids = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        filt_labels = []
        for label in labels:
            if label_counts[label] >= min_freq:
                filt_labels.append(label)
        if filt_labels:
            filt_x.append(provision)
            filt_y.append(filt_labels)
            filt_doc_ids.append(doc_id)
    return filt_x, filt_y, filt_doc_ids


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

    print('Loading data')
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    x, y, doc_ids = unique_data(x, y, doc_ids)
    y = split_conjuncted_labels(y)
    y = merge_plural_label_names(y)
    x, y, doc_ids = remove_lowfreq_labels(x, y, doc_ids)

    outlier_labels = get_outlier_labels(y, doc_ids, do_plot=True)
    x, y, doc_ids = remove_labels(x, y, doc_ids, outlier_labels)

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_clean.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(x, y, doc_ids):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')



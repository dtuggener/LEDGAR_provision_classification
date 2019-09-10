import json
import networkx as nx
from typing import List
from nltk.corpus import stopwords
from collections import defaultdict, Counter


def get_label_hierarchy(y):

    def get_ngrams(words):
        for i in range(len(words) + 1):
            for j in range(i, len(words[i:]) + 1):
                ngram = words[i:i + j]
                if ngram:
                    yield tuple(ngram)

    # TODO stemming, or at least remove plural-S from label words;
    #  better: do full lemmatization (spaCy)?
    stop_words = set(stopwords.words('english'))

    label_list = list(set([l for labels in y for l in labels]))
    label_words_list = []
    for label in label_list:
        words = tuple(sorted([w for w in label.split(' ') if w not in stop_words]))
        label_words_list.append(words)
    label_words_set = set(label_words_list)

    ngram_counts = Counter()
    for words in label_words_list:
        for ngram in get_ngrams(words):
            # if ngram in label_words_set:  # Only take those that are actual labels?
            ngram_counts[ngram] += 1

    # Prune ngrams with frequency 1, they don't have children
    ngram_counts_pruned = Counter({ngram: cnt for ngram, cnt in ngram_counts.most_common() if cnt > 1})

    g = nx.DiGraph()
    # For each label, get its parents, i.e. from ngram_counts_pruned
    for ix, label_words in enumerate(label_words_list):
        if len(label_words) > 1:
            label_parents = Counter()
            for ngram in get_ngrams(label_words):
                if ngram_counts_pruned[ngram] > 0:
                    label_parents[ngram] = ngram_counts_pruned[ngram]
            breakpoint()
            # TODO populate graph;
            #  add parents with least counts / longest label name as parent; add others as grandparents?
        else:
            g.add_node(label_words[0])

    real_labels = dict()
    for node in g.nodes():
        real_labels[node] = True if node in label_words_list else False
    nx.set_node_attributes(g, 'real_label', real_labels)

    # TODO: find labels that don't have multiple parents -> merge??!!!
    # TODO: allow splitting of lowfreq label names into sufficiently frequent constituents;
    #  e.g. 'violation of environmental laws' -> 'violation'; 'environmental laws'


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean_freq100.jsonl'

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data')
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    get_label_hierarchy(y)
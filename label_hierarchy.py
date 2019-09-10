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
    map2original_label = dict()
    for label in label_list:
        # represent label words as sorted bag-of-words tuples to store as keys
        words = tuple(sorted([w for w in label.split(' ') if w not in stop_words]))
        label_words_list.append(words)
        map2original_label[words] = label

    ngram_counts = Counter()
    for words in label_words_list:
        for ngram in get_ngrams(words):
            ngram_counts[ngram] += 1

    g = nx.DiGraph()
    ngrams = sorted(ngram_counts.keys(), key=len, reverse=True)
    # Start with long ngrams, find parent
    for i, ngram in enumerate(ngrams):
        for ngram2 in ngrams[i+1:]:
            if len(ngram2) < len(ngram):  # Parent can't have longer name than child
                if not any(w for w in ngram2 if w not in ngram):  # Child contains all words of parent
                    # TODO this leads to funky business, bug fix it before uncommenting
                    # ngram_node = map2original_label.get(ngram, ngram)
                    # ngram2_node = map2original_label.get(ngram2, ngram2)
                    g.add_edge(ngram, ngram2)

    real_labels = dict()
    for node in g.nodes():
        real_labels[node] = True if node in map2original_label else False
        # real_labels[node] = True if node in map2original_label.values() else False
    nx.set_node_attributes(g, real_labels, 'real_label')
    nx.set_node_attributes(g, ngram_counts, 'weight')
    breakpoint()
    nx.write_gexf(g, 'label_hierarchy.gexf')
    # TODO prune the tree, i.e. remove path parts consisting of non-real labels only;
    #  check for all parents if they share a parent with the child, then remove edge between child and parent's parent
    # access immediate children
    # g.predecessors(('environmental','laws'))
    # access immediate parent
    # g.successors(('environmental','laws'))

    """
    # Prune ngrams with frequency 1, they don't have children
    ngram_counts_pruned = Counter({ngram: cnt for ngram, cnt in ngram_counts.most_common() if cnt > 1})
    
    # For each label, get its parents, i.e. from ngram_counts_pruned
    for ix, label_words in enumerate(label_words_list):
        if len(label_words) > 1:
            compounds = Counter()
            for ngram in get_ngrams(label_words):
                if ngram_counts_pruned[ngram] > 0:
                    compounds[ngram] = ngram_counts_pruned[ngram]
            print(label_words, compounds)
            breakpoint()
            # TODO populate graph;
            #  add parents with least counts / longest label name as parent; add others as grandparents?
        else:
            g.add_node(label_words[0])
    """

    # TODO: find labels that don't have multiple parents -> merge??!!!
    # TODO: allow splitting of lowfreq label names into sufficiently frequent constituents;
    #  e.g. 'violation of environmental laws' -> 'violation'; 'environmental laws'


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean_freq100.jsonl'
    print('Loading data from', corpus_file)

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    get_label_hierarchy(y)
import json
import networkx as nx
from typing import List, Set, Tuple
from nltk.corpus import stopwords
from collections import defaultdict, Counter


def tuple_contains(tup1: Tuple, tup2: Tuple) -> Tuple[bool, int]:
    """Check whether tuple 1 contains tuple 2"""
    len_tup1, len_tup2 = len(tup1), len(tup2)
    for i in range(0, len_tup1 + 1 - len_tup2):
        if tup1[i:i + len_tup2] == tup2:
            return True, i
    return False, -1


def get_ngrams(words):
    for i in range(len(words) + 1):
        for j in range(i, len(words[i:]) + 1):
            ngram = words[i:i + j]
            if ngram:
                yield tuple(ngram)


def get_base_forms(label_set: Set[str]):
    """Determine singular forms"""
    token_set = set([l for label in label_set for l in label.split(' ')])
    base_forms = dict()
    for token in token_set:
        if token.endswith('s') and token[:-1] in token_set:
            base_forms[token] = token[:-1]
        elif token.endswith('ies') and token[:-3] + 'y' in token_set:
            base_forms[token] = token[:-3] + 'y'
    return base_forms


def real_label_hierarchy_graph(y) -> nx.DiGraph:

    print('Getting token baseforms')
    label_list = list(set([l for labels in y for l in labels]))
    base_forms = get_base_forms(set(label_list))

    print('Lemmatizing labels')
    label_set_lemmas = set()
    label2lemma = dict()

    for label in label_list:
        lemmas = tuple([base_forms.get(w, w) for w in label.split(' ')])
        label2lemma[label] = lemmas
        label_set_lemmas.add(lemmas)

    y_lemmas = []
    stop_words = set(stopwords.words('english'))
    for labels in y:
        labels_lemmas = []
        for label in labels:
            label_words = label.split(' ')
            filtered = [l for l in label_words if l not in stop_words]
            if filtered:
                labels_lemmas.append(label2lemma[label])
        y_lemmas.append(set(labels_lemmas))

    label_counts = Counter([l for labels in y_lemmas for l in labels])
    print('Populating graph based on {} labels'.format(len(label_counts)))
    g = nx.DiGraph()
    label_lemmas = sorted(label_counts.keys(), key=len, reverse=True)
    label_lemmas_by_lengths = defaultdict(list)
    for label in label_lemmas:  # Bucket ngrams by lengths for faster comparison
        label_lemmas_by_lengths[len(label)].append(label)
    sorted_lengths = sorted(label_lemmas_by_lengths.keys(), reverse=True)

    proc_cnt = 0
    for i, length in enumerate(sorted_lengths):
        for label in label_lemmas_by_lengths[length]:
            proc_cnt += 1
            print(str(proc_cnt) + '\r', end='', flush=True)
            for length2 in sorted_lengths[i+1:]:
                for label2 in label_lemmas_by_lengths[length2]:
                    if tuple_contains(label, label2)[0]:
                            g.add_edge(label, label2)

    nx.set_node_attributes(g, label_counts, 'weight')
    nx.set_node_attributes(g, {l: True for l in label_counts}, 'real_label')
    return g


def label_hierarchy_graph(y) -> nx.DiGraph:

    print('Getting token baseforms')
    label_list = list(set([l for labels in y for l in labels]))
    base_forms = get_base_forms(set(label_list))

    print('Lemmatizing labels and counting ngrams')
    ngram_counts = Counter()
    label_set_lemmas = set()
    label2lemma = dict()
    stop_words = set(stopwords.words('english'))
    for label in label_list:
        lemmas = tuple([base_forms.get(w, w) for w in label.split(' ')])
        label2lemma[label] = lemmas
        label_set_lemmas.add(lemmas)
        for ngram in get_ngrams(lemmas):
            if ngram == label:
                ngram_counts[ngram] += 1
            else:
                # Filter ngrams that consist of stopwords only, or those that have stop words at borders
                filtered = [l for l in ngram if l not in stop_words]
                if filtered:
                    if not ngram[-1] in stop_words and not ngram[0] in stop_words:
                        ngram_counts[ngram] += 1
    print('Found', len(ngram_counts), 'ngrams')

    print('Populating graph')
    g = nx.DiGraph()
    ngrams = sorted(ngram_counts.keys(), key=len, reverse=True)
    ngrams_by_lengths = defaultdict(list)
    for ngram in ngrams:  # Bucket ngrams by lengths for faster comparison
        ngrams_by_lengths[len(ngram)].append(ngram)
    sorted_lengths_ngrams = sorted(ngrams_by_lengths.keys(), reverse=True)

    proc_cnt = 0
    for i, length in enumerate(sorted_lengths_ngrams):
        for ngram in ngrams_by_lengths[length]:
            proc_cnt += 1
            print(str(proc_cnt) + '\r', end='', flush=True)
            for length2 in sorted_lengths_ngrams[i+1:]:
                for ngram2 in ngrams_by_lengths[length2]:
                    if tuple_contains(ngram, ngram2)[0]:
                            g.add_edge(ngram, ngram2)
                            break

    real_labels = {l: True if l in label_set_lemmas else False for l in ngram_counts.keys()}
    nx.set_node_attributes(g, real_labels, 'real_label')
    label_counts = Counter(l for labels in y for l in labels)
    label_counts_lemmas = {label2lemma[l]: c for l, c in label_counts.items()}
    nx.set_node_attributes(g, label_counts_lemmas, 'weight')
    return g


def prune_real_graph(g: nx.DiGraph) -> nx.DiGraph:
    while True:
        old_edge_count, old_node_count = len(g.edges()), len(g.nodes())
        # Remove edges to grandparents
        del_edges = []
        for node in g.nbunch_iter():
            neighbors = list(g.successors(node))
            for neighbor in neighbors:
                # neighbor_neighbors = list(g.successors(neighbor))
                neighbor_descendants = nx.descendants(g, neighbor)
                shared_neighbors = [n for n in neighbor_descendants if n in neighbors]
                if shared_neighbors:
                    # Remove edges from node to shared neighbors
                    for shared_neighbor in shared_neighbors:
                        del_edges.append((node, shared_neighbor))
        g.remove_edges_from(del_edges)
        if len(g.edges()) == old_edge_count and len(g.nodes()) == old_node_count:
            break
    return g


def prune_graph(g: nx.DiGraph) -> nx.DiGraph:
    while True:
        old_edge_count, old_node_count = len(g.edges()), len(g.nodes())
        # Remove edges to grandparents
        del_edges = []
        for node in g.nbunch_iter():
            neighbors = list(g.successors(node))
            for neighbor in neighbors:
                # neighbor_neighbors = list(g.successors(neighbor))
                neighbor_descendants = nx.descendants(g, neighbor)
                shared_neighbors = [n for n in neighbor_descendants if n in neighbors]
                if shared_neighbors:
                    # Remove edges from node to shared neighbors
                    for shared_neighbor in shared_neighbors:
                        del_edges.append((node, shared_neighbor))
        g.remove_edges_from(del_edges)

        # Remove synthetic nodes with only one predecessor;
        # link predecessor to successors directly
        single_successors_synthetic_nodes = [n for n in g.nbunch_iter() if len(list(g.predecessors(n))) == 1
                                             and not g.nodes()[n]['real_label']]
        for node in single_successors_synthetic_nodes:
            child = list(g.predecessors(node))[0]
            parents = list(g.successors(node))
            if parents:
                for parent in parents:
                    g.add_edge(child, parent)
            g.remove_node(node)

        if len(g.edges()) == old_edge_count and len(g.nodes()) == old_node_count:
            break
    return g


def add_ancestor_support(g):
    for node in g.nodes():
        ancestor_support = sum([g.nodes()[anc].get('weight', 0) for anc in nx.ancestors(graph, node)])
        g.nodes()[node]['ancestor support'] = ancestor_support
    return g


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
    print('Loading data from', corpus_file)

    y: List[List[str]] = []

    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        y.append(labeled_provision['label'])

    graph = real_label_hierarchy_graph(y)
    graph = prune_graph(graph)

    graph = add_ancestor_support(graph)

    nx.write_gexf(graph, corpus_file.replace('.jsonl', '_label_hierarchy.gexf'))


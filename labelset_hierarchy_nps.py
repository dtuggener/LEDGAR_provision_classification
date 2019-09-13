import json
import networkx as nx
from typing import List, Set
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import spacy

nlp = spacy.load("en_core_web_sm")


def labels2graph(y):
    g = nx.DiGraph()
    label_counts = Counter({l for labels in y for l in labels})
    nlp = spacy.load("en_core_web_sm")
    for label, cnt in label_counts.items():
        parse = nlp(label)
        if len(parse) > 1:
            print(label)
            # just serialize and iteratively add each token as parent node; resulting in an path to the head noun?
            for token in parse:
                print(token.text, token.lemma_, token.dep_, token.pos_)
                """
                for nchunk in parse.noun_chunks:
                    print(nchunk, nchunk.root.text)
                    for token in nchunk:
                        print(token, token.dep_)
                """
            breakpoint()
        else:
            g.add_node(label, weight=label_counts[label])


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


def label_hierarchy_graph(y) -> nx.DiGraph:

    print('Getting token baseforms')
    label_list = list(set([l for labels in y for l in labels]))
    base_forms = get_base_forms(set(label_list))

    print('Lemmatizing labels and counting ngrams')
    ngram_counts = Counter()
    label_set_lemmas = set()
    for label in label_list:
        lemmas = tuple([base_forms.get(w, w) for w in label.split(' ')])
        label_set_lemmas.add(lemmas)
        for ngram in get_ngrams(lemmas):
            ngram_counts[ngram] += 1

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
            len_ngram = len(ngram)
            for length2 in sorted_lengths_ngrams[i+1:]:
                for ngram2 in ngrams_by_lengths[length2]:
                    len_ngram2 = len(ngram2)
                    for j in range(0, len_ngram + 1 - len_ngram2):
                        if ngram[j:j+len_ngram2] == ngram2:
                            g.add_edge(ngram, ngram2)

    real_labels = {l: True if l in label_set_lemmas else False for l in ngram_counts.keys()}
    nx.set_node_attributes(g, real_labels, 'real_label')
    nx.set_node_attributes(g, ngram_counts, 'weight')
    return g


def prune_graph(g: nx.DiGraph) -> nx.DiGraph:
    while True:
        old_edge_count, old_node_count = len(g.edges()), len(g.nodes())
        # Remove edges to grandparents
        del_edges = []
        for node in g.nbunch_iter():
            neighbors = list(g.successors(node))
            for neighbor in neighbors:
                neighbor_neighbors = list(g.successors(neighbor))
                shared_neighbors = [n for n in neighbor_neighbors if n in neighbors]
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

        # TODO prune preposition nodes!!

        if len(g.edges()) == old_edge_count and len(g.nodes()) == old_node_count:
            break
    return g


if __name__ == '__main__':

    # corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
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

    graph = label_hierarchy_graph(y)
    graph_pruned = prune_graph(graph)
    nx.write_gexf(graph, corpus_file.replace('.jsonl', '_label_hierarchy.gexf'))
    breakpoint()
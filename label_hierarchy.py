import json
import networkx as nx
from typing import List
from nltk.corpus import stopwords
from collections import defaultdict, Counter


def get_ngrams(words):
    for i in range(len(words) + 1):
        for j in range(i, len(words[i:]) + 1):
            ngram = words[i:i + j]
            if ngram:
                yield tuple(ngram)


def label_hierarchy_graph(y) -> nx.DiGraph:

    label_list = list(set([l for labels in y for l in labels]))

    # Determine singular forms
    token_set = set([l for label in label_list for l in label.split(' ')])
    base_forms = dict()
    for token in token_set:
        if token.endswith('s') and token[:-1] in token_set:
            base_forms[token] = token[:-1]
        elif token.endswith('ies') and token[:-3] + 'y' in token_set:
            base_forms[token] = token[:-3] + 'y'

    label_words_list = []
    tuple2label = dict()
    stop_words = set(stopwords.words('english'))
    for label in label_list:
        # represent label words as sorted bag-of-words tuples to store as keys
        lemmas = [base_forms.get(w, w) for w in label.split(' ')]
        lemmas = tuple(sorted([w for w in lemmas if w not in stop_words]))
        # words = tuple(sorted([w for w in label.split(' ') if w not in stop_words]))
        label_words_list.append(lemmas)
        tuple2label[lemmas] = label

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
                    # g.add_edge(ngram, ngram2)
                    g.add_edge(tuple2label.get(ngram, ngram), tuple2label.get(ngram2, ngram2))

    # Tag nodes wrt to source, either observed or synthetic labels
    real_labels = dict()
    label2tuple = Counter({l: t for t, l in tuple2label.items()})
    for node in g.nodes():
        real_labels[node] = True if node in label2tuple else False
    nx.set_node_attributes(g, real_labels, 'real_label')
    nx.set_node_attributes(g, ngram_counts, 'weight')

    # Add frequency of (synthetic) labels as weights
    label_counts = Counter([l for labels in y for l in labels])
    # tuple_counts = {label2tuple[l]: c for l, c in label_counts.items()}
    nx.set_node_attributes(g, label_counts, 'weight')

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
        if len(g.edges()) == old_edge_count and len(g.nodes()) == old_node_count:
            break
    return g


def create_subgraph(graph: nx.DiGraph):
    root_node = ('request',)
    root_node2 = ('borrowing',)
    children = nx.ancestors(graph, root_node)
    children.add(root_node)
    children2 = nx.ancestors(graph, root_node2)
    children2.add(root_node2)
    all_children = children.union(children2)
    sg = nx.subgraph(graph, all_children)
    nx.write_gexf(sg, 'label_hierarchy_sg.gexf')
    """
    import matplotlib.pyplot as plt
    nx.draw_networkx(sg)
    plt.show()
    """


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
    graph = prune_graph(graph)
    nx.write_gexf(graph, 'label_hierarchy.gexf')
    roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n))]
    real_roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n)) and graph.nodes()[n]['real_label']]
    breakpoint()

    # TODO: allow splitting of lowfreq label names (f<25 or f<50) into sufficiently frequent constituents;
    #  e.g. 'violation of environmental law' -> 'violation'; 'environmental law'
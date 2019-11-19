import json
import networkx as nx
import numpy
from typing import Set, Dict, List, Tuple
from labelset_hierarchy import get_base_forms


def create_subgraph(graph: nx.DiGraph, root_node, direction='descendants'):
    if direction == 'descendants':
        children = nx.descendants(graph, root_node)
    else:
        children = nx.ancestors(graph, root_node)
    children.add(root_node)
    sg = nx.subgraph(graph, children)
    nx.write_gexf(sg, '/tmp/label_hierarchy_sg.gexf')
    return sg


def find_lowfreq_hubs(g):
    node_anc_weight = []
    for node in g.nodes():
        ancestors = list(nx.ancestors(graph, node))
        if ancestors:
            avg_anc_weights  = numpy.mean([g.nodes()[anc].get('weight', 0) for anc in ancestors])
            if avg_anc_weights > 0:
                node_anc_weight.append((avg_anc_weights, node))
    node_anc_weight.sort()
    breakpoint()


def get_popular_descendants(node, g, descendants=None, min_freq=50):
    if descendants is None:
        descendants = set()
    successors = list(g.successors(node))
    if successors:
        for neighbor in successors:
            if g.nodes()[neighbor].get('weight', 0) >= min_freq or  \
                    g.nodes()[neighbor].get('ancestor support', 0) >= min_freq:
                descendants.add(neighbor)
            else:
                return get_popular_descendants(neighbor, g, descendants=descendants, min_freq=min_freq)
    return descendants


def map_lowfreq_labels(g: nx.DiGraph, min_freq: int = 50) -> Dict[str, Set[str]]:
    label_merges = dict()
    for node in g.nbunch_iter():

        if g.nodes()[node]['real_label'] and \
                g.nodes()[node].get('weight', 0) < min_freq and \
                g.nodes()[node].get('ancestor support', 0) < min_freq and \
                len(node) > 1:
            scored_neighbors = []
            mapped_labels = set()

            for neighbor in g.successors(node):
                neighbor_weight = g.nodes()[neighbor].get('weight', 0)
                scored_neighbors.append((neighbor_weight, neighbor))
            scored_neighbors.sort(reverse=True)

            for score, neighbor in scored_neighbors:
                if (score >= min_freq or
                        g.nodes()[neighbor].get('ancestor support', 0) >= min_freq):  # and g.nodes()[neighbor]['real_label']:  # Allow synthetic labels?
                    mapped_labels.add(neighbor)
                else:
                    descendants = get_popular_descendants(neighbor, g)
                    mapped_labels.update(descendants)

            label_merges[node] = mapped_labels

    return label_merges


def decompose_to_roots(g: nx.DiGraph) -> Dict[str, List[str]]:
    label2roots = dict()
    roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n)) and
             list(graph.predecessors(n))]
    for node in g:
        if len(node) > 1:
            descendants = nx.descendants(g, node)
            if descendants:
                root_labels = [' '.join(l) for l in descendants if l in roots]
            else:
                root_labels = [' '.join(node)]
            print(node)
            print(root_labels)
            breakpoint()
        else:
            root_labels = [' '.join(node)]
        label2roots[' '.join(node)] = root_labels

    return label2roots


def decompose_real_labels_to_roots(g: nx.DiGraph) -> Dict[str, List[str]]:
    label2roots = dict()
    for node in g:
        if len(node) > 1:
            descendants = nx.descendants(g, node)
            if descendants:
                real_root_labels = [' '.join(l) for l in descendants if not list(g.successors(l))]
            else:
                real_root_labels = [' '.join(node)]
        else:
            real_root_labels = [' '.join(node)]
        label2roots[' '.join(node)] = real_root_labels
    return label2roots


def prune_sparse_roots(g: nx.DiGraph, min_freq: int = 50) -> Tuple[nx.DiGraph, List[Tuple[str]]]:
    spare_roots = [n for n in g.nodes() if not list(g.successors(n)) and
                   g.nodes()[n].get('weight', 0) < min_freq and
                   g.nodes()[n].get('ancestor support', 0) < min_freq]
    g.remove_nodes_from(spare_roots)
    return g, spare_roots


if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
    graph_file = corpus_file.replace('.jsonl', '_real_label_hierarchy.gexf')
    print('Reading graph from', graph_file)
    graph = nx.read_gexf(graph_file)

    # Convert node names from strings back to tuples:
    name_map = {l: eval(l) for l in graph.nodes()}
    graph = nx.relabel_nodes(graph, name_map)

    graph, sparse_roots = prune_sparse_roots(graph)
    sparse_roots = {' '.join(l) for l in sparse_roots}

    # Split labels into parents with sufficient support
    #label_merges = map_lowfreq_labels(graph, min_freq=100)
    #label_set_size = len(set([l for labels in label_merges.values() for l in labels]))
    #breakpoint()

    # Decompose into (real) roots
    label_merges = decompose_real_labels_to_roots(graph)

    print('Loading data from', corpus_file)
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    label_set = set(l for labels in y for l in labels)
    base_forms = get_base_forms(label_set)

    new_y, new_x, new_doc_ids = [], [], []
    dumped_labels = set()
    for x_i, y_i, doc_id in zip(x, y, doc_ids):
        new_y_i: List[str] = list()
        for label in y_i:
            label = ' '.join(base_forms.get(l, l) for l in label.split())
            if label not in sparse_roots and label in label_merges:
                new_y_i.extend(label_merges[label])
            else:
                dumped_labels.add(label)
        if new_y_i:
            new_x.append(x_i)
            new_doc_ids.append(doc_id)
            new_y.append(new_y_i)

    x, y, doc_ids = new_x, new_y, new_doc_ids
    label_set = set([l for labels in y for l in labels])
    breakpoint()

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_projected_real_roots.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(x, y, doc_ids):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')


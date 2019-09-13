import networkx as nx
import numpy


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


if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
    graph_file = corpus_file.replace('.jsonl', '_label_hierarchy.gexf')
    graph = nx.read_gexf(graph_file)

    roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n))]
    real_roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n)) and graph.nodes()[n]['real_label']]

    # find nodes where the average weight of the ancestors is low
    find_lowfreq_hubs(graph)

    min_freq = 50

    lowfreq_labels = [n for n in graph if 0 < graph.nodes()[n].get('weight', 0) < 50 and graph.nodes()[n]['real_label']]

    sg = create_subgraph(graph, lowfreq_labels[0])

    breakpoint()

    # TODO: allow splitting of lowfreq label names (f<25 or f<50) into sufficiently frequent constituents;
    #  e.g. 'violation of environmental law' -> 'violation'; 'environmental law'

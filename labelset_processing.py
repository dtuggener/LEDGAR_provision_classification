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


def map_lowfreq_labels(g: nx.DiGraph, min_freq: int = 50):
    label_merges = {}
    for node in g.nodes():
        if g.nodes()[node]['real_label'] and g.nodes()[node].get('weight', 0) < min_freq:
            # TODO find descending neighbor with either most support or most ancestor support
            scored_neighbors = []
            for neighbor in nx.neighbors(g, node):
                neighbor_weight = g.nodes()[neighbor].get('weight', 0)
                scored_neighbors.append((neighbor_weight, neighbor))
            scored_neighbors.sort(reverse=True)
            print(node)
            print(scored_neighbors)
            breakpoint()


if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
    graph_file = corpus_file.replace('.jsonl', '_label_hierarchy.gexf')
    graph = nx.read_gexf(graph_file)

    roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n))]
    real_roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n)) and graph.nodes()[n]['real_label']]

    map_lowfreq_labels(graph)

    # find nodes where the average weight of the ancestors is low
    find_lowfreq_hubs(graph)

    min_freq = 50

    lowfreq_labels = [n for n in graph if 0 < graph.nodes()[n].get('weight', 0) < 50 and graph.nodes()[n]['real_label']]

    sg = create_subgraph(graph, lowfreq_labels[0])

    # TODO
    """
    interesting:
('termination', 'by', 'tyson', 'without', 'cause', 'or', 'by', 'you', 'for', 'good', 'reason')
[(122, "('good', 'reason')"), (9, "('termination', 'by', 'tyson', 'without', 'cause')")]
print(list(g.successors("('termination', 'by', 'tyson', 'without', 'cause')")))
["('termination',)", "('without', 'cause')"]
-> 'by tyson' gets croped out, exactly as we want!
-> interesting: we have a node ('termination', 'without', 'cause');
i.e. we could check if a target has a common ancestors/if the concatenation of the labels is a (true) label! if yes, take that as the merge target!
     """

    breakpoint()
    # TODO find descending neighbor with either most support, or most ancestor support

    # TODO check for node labels that consist of token with strong association
    #  (i.e. "change of control"; "governing law")
    # TODO: allow splitting of lowfreq label names (f<25 or f<50) into sufficiently frequent constituents;
    #  e.g. 'violation of environmental law' -> 'violation'; 'environmental law'

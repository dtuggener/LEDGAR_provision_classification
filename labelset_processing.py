import networkx as nx

if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean_freq100.jsonl'
    graph_file = corpus_file.replace('.jsonl', '_label_hierarchy.gexf')
    graph = nx.read_gexf(graph_file)

    roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n))]
    real_roots = [n for n in graph.nbunch_iter() if not list(graph.successors(n)) and graph.nodes()[n]['real_label']]
    min_freq = 50

    lowfreq_labels = [n for n in graph if graph.nodes()[n]['weight'] < 50 and graph.nodes()[n]['real_label']]

    breakpoint()

    # TODO: allow splitting of lowfreq label names (f<25 or f<50) into sufficiently frequent constituents;
    #  e.g. 'violation of environmental law' -> 'violation'; 'environmental law'

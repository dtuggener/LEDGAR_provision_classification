import json
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def label_stats(x, y, doc_ids, n=10):
    labels = [l for labels in y for l in labels]
    label_counts = Counter(labels)
    print(len(label_counts), 'labels,', len(x), 'provisions', len(set(doc_ids)), 'contracts')
    for label, cnt in label_counts.most_common(n):
        print(label, cnt)
    ml = [(text, labels) for text, labels in zip(x, y) if len(labels) > 1]
    print('{} provisions with multilabels ({}%)'.format(len(ml), round(100*len(ml)/len(y), 2)))


def provision_type_similarity(vecs_per_label, do_plot=False):
    """Pairwise similarity between averaged tf idf vectors of provisions sharing a label"""
    label_set = list(set(vecs_per_label.keys()))
    breakpoint()
    avg_vecs = [numpy.mean(vecs_per_label[label], axis=0) for label in label_set]
    similar_labels = []
    dists = squareform(pdist(avg_vecs))
    for i, (label, label_dist) in enumerate(zip(label_set, dists)):
        for j, label2 in enumerate(label_set[i+1:]):
            similar_labels.append((label_dist[i+1+j], label, label2))
    similar_labels.sort()
    if do_plot:
        from scipy.cluster.hierarchy import linkage, dendrogram
        hcluster = linkage(avg_vecs, method='ward')
        dendrogram(hcluster, orientation='right', labels=label_set)
        plt.savefig('/tmp/label_similarity_dengrogram.png')
    return similar_labels


def sample_frequent_labels(x, y, doc_ids, min_freq=None, max_freq=None, n_labels=None):
    label_counts = Counter([l for labels in y for l in labels])
    selected_labels = label_counts.copy()

    if min_freq:
        selected_labels = Counter({l: c for l, c in selected_labels.items() if c >= min_freq})
    if max_freq:
        selected_labels = Counter({l: c for l, c in selected_labels.items() if c <= max_freq})
    if n_labels:
        selected_labels = Counter({l: c for (l, c) in label_counts.most_common(n_labels)})

    x_small, y_small, doc_ids_small = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        sel_labels = [l for l in labels if l in selected_labels]
        if sel_labels:
            x_small.append(provision)
            y_small.append(sel_labels)
            doc_ids_small.append(doc_id)
    return x_small, y_small, doc_ids_small


def sample_common_labels(x, y, doc_ids, n_labels=20):
    labels2docs = defaultdict(set)
    for provision, labels, doc_id in zip(x, y, doc_ids):
        for label in labels:
            labels2docs[label].add(doc_id)
    labels2docs_counts = Counter({l: len(ds) for l, ds in labels2docs.items()})
    selected_labels = [l for (l, _) in labels2docs_counts.most_common(n_labels)]

    filt_x, filt_y, filt_doc_ids = [], [], []
    for provision, labels, doc_id in zip(x, y, doc_ids):
        filt_labels = []
        for label in labels:
            if label in selected_labels:
                filt_labels.append(label)
        if filt_labels:
            filt_x.append(provision)
            filt_y.append(filt_labels)
            filt_doc_ids.append(doc_id)
    return filt_x, filt_y, filt_doc_ids


def avg_provision_count(y, doc_ids):
    doc2labels = defaultdict(list)
    for labels, doc_id in zip(y, doc_ids):
        doc2labels[doc_id].append(labels)
    doc2labels_counts = Counter({doc_id: len(labels) for doc_id, labels in doc2labels.items()})
    avg_prov_count = int(numpy.mean(list(doc2labels_counts.values())))
    return avg_prov_count


def plot_label_pca_means(x_tfidf, y):
    colors = list(mcolors.CSS4_COLORS.keys())
    label2color = {l: c for l, c in zip(y, colors)}
    pca = PCA(n_components=2).fit_transform(x_tfidf)
    for vec, label in zip(pca, y):
        plt.scatter(vec[0], vec[1], label=label, c=label2color[label])
    plt.legend()
    breakpoint()


def get_provision_diversity(x_tfidf: numpy.array, y: List[List[str]]):
    div = []
    labels2vecs = defaultdict(list)
    for vec, labels in zip(x_tfidf, y):
        if len(labels) > 1:
            continue  # omit multilabels
        labels2vecs[labels[0]].append(vec)
    breakpoint()
    for label, vecs in labels2vecs.items():
        vecs = [v.toarray()[0] for v in vecs]
        avg_dist = numpy.mean(pdist(numpy.array(vecs), metric='cosine'))
        div.append((avg_dist, label))
    div.sort()
    return div


def remove_stopwords(x):
    x_filt = []
    stopWords = set(stopwords.words('english'))
    for provision in x:
        filt_toks = [w for w in word_tokenize(provision) if w.lower() not in stopWords]
        x_filt.append(' '.join(filt_toks))
    return x_filt


def write_jsonl(out_file: str, x_small, y_small, doc_ids_small):
    print('Writing output')
    with open(out_file, 'w', encoding='utf8') as f:
        for provision, labels, doc_id in zip(x_small, y_small, doc_ids_small):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')


def create_subcorpora(x, y, doc_ids):
    print('Sampling most common provisions')
    avg_prov_cnt = avg_provision_count(y, doc_ids) # Average no. of provisions per contract
    x_small, y_small, doc_ids_small = sample_common_labels(x, y, doc_ids, n_labels=avg_prov_cnt)
    label_stats(x_small, y_small, doc_ids_small)
    out_file  = corpus_file.replace('.jsonl', '_proto.jsonl')
    write_jsonl(out_file, x_small, y_small, doc_ids_small)

    print('Sampling provisions with frequency >= 100')
    x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, min_freq=100)
    label_stats(x_small, y_small, doc_ids_small)
    out_file = corpus_file.replace('.jsonl', '_freq100.jsonl')
    write_jsonl(out_file, x_small, y_small, doc_ids_small)

    print('Sampling sparse labels with 10 <= frequency <= 20')
    x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, min_freq=10, max_freq=20)
    label_stats(x_small, y_small, doc_ids_small)
    out_file = corpus_file.replace('.jsonl', '_sparse.jsonl')
    write_jsonl(out_file, x_small, y_small, doc_ids_small)


def incremental_label_stats(x, y, doc_ids):
    for i in [0, 10, 50, 100, 500, 1000, 5000, 10000]:
        print(i)
        x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, min_freq=i)
        label_stats(x_small, y_small, doc_ids_small, n=0)


def label_cooc(y, doc_ids):
    labels2docs = defaultdict(set)
    for labels, doc_id in zip(y, doc_ids):
        for label in labels:
            labels2docs[label].add(doc_id)
    label_list = list(labels2docs.keys())
    similarities = []
    for i, l1 in enumerate(label_list):
        print('\r', i, end='', flush=True)
        l1_docs = labels2docs[l1]
        for l2 in label_list[i+1:]:
            l2_docs = labels2docs[l2]
            jacc_sim = len(l1_docs.intersection(l2_docs)) / len(l1_docs.union(l2_docs))
            if jacc_sim > 0:
                similarities.append((jacc_sim, l1, l2))
    breakpoint()
    similarities.sort(reverse=True)


def plot_label_name_vs_freq(y):
    label_list = [l for labels in y for l in labels]
    label_counts_counter = Counter(label_list)
    name_lengths = []
    label_counts = []
    for label, cnt in label_counts_counter.most_common():
        label_counts.append(cnt)
        name_lengths.append(label.count(' ') + 1)
    plt.scatter(label_counts, name_lengths, marker='+', c='black')
    plt.xlabel('Label frequency')
    plt.ylabel('Label name token count')
    plt.savefig('label_name_length_vs_freq.pdf')


def cluster_labels(y, doc_ids):
    doc2labels = defaultdict(set)
    labels2docs = defaultdict(set)
    for labels, doc_id in zip(y, doc_ids):
        doc2labels[doc_id].update(labels)
        for label in labels:
            labels2docs[label].add(doc_id)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
    cv = CountVectorizer(binary=True, tokenizer=lambda x: x, preprocessor=lambda x: x)

    label_vecs = cv.fit_transform(list(labels2docs.values()))
    label_list = list(labels2docs.keys())
    # TODO maybe softclustering is more appropriate here
    kmeans = KMeans(n_clusters=50).fit(label_vecs)
    clusters = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(label_list[i])
    breakpoint()


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data from', corpus_file)
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    labels = [l for labels in y for l in labels]
    label_counts = Counter(labels)

    create_subcorpora(x, y, doc_ids)

    # Find label clusters that often occur together in documents
    # cluster_labels(y, doc_ids)

    # label_stats(x, y, doc_ids, n=0)

    # plot_label_name_vs_freq(y)

    # incremental_label_stats(x, y, doc_ids)

    # similar_labels = provision_type_similarity(vecs_per_label)

    # label_cooc(y, doc_ids)


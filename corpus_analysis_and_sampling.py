import json
import pdb
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def label_stats(x, y, n=100):
    labels = [l for labels in y for l in labels]
    label_counts = Counter(labels)
    print(len(label_counts), 'labels')
    for label, cnt in label_counts.most_common(n):
        print(label, cnt)
    ml = [(text, labels) for text, labels in zip(x, y) if len(labels) > 1]
    print('{} provisions with multilabels ({}%)'.format(len(ml), round(100*len(ml)/len(y), 2)))
    pdb.set_trace()


def provision_type_similarity(vecs_per_label, do_plot=False):
    """Pairwise similarity between averaged tf idf vectors of provisions sharing a label"""
    label_set = list(set(vecs_per_label.keys()))
    pdb.set_trace()
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


def sample_frequent_labels(x, y, doc_ids, min_freq=None, n_labels=None):
    label_counts = Counter([l for labels in y for l in labels])

    if min_freq:
        selected_labels = {l for l, c in label_counts.items() if c >= min_freq}
    elif n_labels:
        selected_labels = {l for (l, _) in label_counts.most_common(n_labels)}

    assert selected_labels, 'No label frequency criteria defined'

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
    pdb.set_trace()


def get_provision_diversity(x_tfidf: numpy.array, y: List[List[str]]):
    div = []
    labels2vecs = defaultdict(list)
    for vec, labels in zip(x_tfidf, y):
        if len(labels) > 1:
            continue  # omit multilabels
        labels2vecs[labels[0]].append(vec)
    pdb.set_trace()
    for label, vecs in labels2vecs.items():
        vecs = [v.toarray()[0] for v in vecs]
        avg_dist = numpy.mean(pdist(vecs, metric='cosine'))
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


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'

    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []

    print('Loading data')
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    label_stats(x, y)

    print('Sampling')
    # Average no. of provisions per contract
    avg_provisions = avg_provision_count(y, doc_ids)

    # Sample most frequent labels
    # x_small, y_small, doc_ids_small = sample_frequent_labels(x, y, doc_ids, n_labels=20)

    # Sample labels that occur in most contracts ~> a general provision inventory
    x_small, y_small, doc_ids_small = sample_common_labels(x, y, doc_ids, n_labels=avg_provisions)

    label_stats(x_small, y_small)

    print('Removing stopwords')
    x_small = remove_stopwords(x_small)

    print('Vectorizing')
    x_small_tfidf = TfidfVectorizer(sublinear_tf=True, max_features=10000).fit_transform(x_small)
    pdb.set_trace()

    # Diversity of selected provisions
    div = get_provision_diversity(x_small_tfidf, y_small)
    pdb.set_trace()

    # plot_label_pca_means(x_vec_means, y_means)

    # TODO
    # similarity of selected provisions
    # avg. length of provisions, no. of labels, no. of provisions per label ...

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_sampled.jsonl'), 'w', encoding='utf8') as f:
        for provision, labels, doc_id in zip(x_small, y_small, doc_ids_small):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')

    """
    
    
    # doesn't work for large data sets (oov
    x_tfidf = TfidfVectorizer(sublinear_tf=True).fit_transform(x).toarray()

    # plot_label_pca(x_tfidf, y)

    vecs_per_label, segs_per_label = defaultdict(list), defaultdict(list)
    for seg, seg_vec, labels in zip(x, x_tfidf, y):
        # skip multilabels
        if len(labels) > 1:
            continue
        label = labels[0]
        vecs_per_label[label].append(seg_vec)
        segs_per_label[label].append(seg)

    div = get_provision_diversity(vecs_per_label)
    pdb.set_trace()

    # similar_labels = provision_type_similarity(vecs_per_label)
    """
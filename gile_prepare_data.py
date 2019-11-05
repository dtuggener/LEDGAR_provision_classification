"""
Convert data to format required by GILE
Code copied and modified from https://github.com/idiap/mhan/blob/master/fetch_data.py
WARNING: HAS TO BE RUN WITH PYTHON2! Doesn't work with python 3+ :(
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import json
import os
import pickle
import gzip


class SplitDataSet:
    def __init__(self, x_train, y_train,
                 x_test, y_test,
                 x_dev=None, y_dev=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_dev = x_dev
        self.y_dev = y_dev


def split_corpus(corpus_file, use_dev = True,
                 test_size = 0.2, dev_size = 0.1,
                 random_state = 42):
    x = []
    y = []
    doc_ids = []
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    if use_dev:
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                          test_size=dev_size,
                                                          random_state=random_state)
    else:
        x_dev, y_dev = None, None

    dataset = SplitDataSet(x_train, y_train, x_test, y_test, x_dev, y_dev)
    return dataset


def clean(text):
    """ Removes special characters from a given text. """
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    return text.replace('\t', ' ')


def extract_wordids(keywords, lang, vocab):
    """ Extracts the word ids for a given set. """
    y_ids = []
    for keyword in keywords:
        keyword = keyword.strip()
        vecs_ids = []
        for word in keyword.split():
            try:
                idy = vocab[lang].index(word)
                vecs_ids.append(idy)
            except:
                continue
        if len(vecs_ids) > 0:
            y_ids.append(vecs_ids)
    return y_ids


def get_label_counts(y_idxs, lang):
    """ Counts the label occurrences in a given set. """
    h = {}
    for y in y_idxs:
        for yy in y:
            key = "_".join([str(yyy.encode('utf8')) for yyy in [yy]])
            if key not in h:
                h[key] = 1
            else:
                h[key] += 1
    return h


def convert_data(x, y, vocab, label_set):
    X, Y = [], []
    for sample, labels in zip(x, y):
        sentences = sent_tokenize(clean(sample.strip()))
        x, x_ids = [], []
        for sentence in sentences:
            vecs, vecs_ids = [], []
            for word in word_tokenize(sentence):
                try:
                    idx = vocab[lang].index(word)
                    vecs_ids.append(idx)
                except:
                    continue
            if len(vecs_ids) > 0:
                x_ids.append(vecs_ids)
        #y_ids = extract_wordids(labels, label_set)
        y_ids = [label_set.index(l) for l in labels]
        X.append(x_ids)
        Y.append(y_ids)
    return X, Y


def load_word_vectors(language, path):
    """Function to load pre-trained word vectors."""
    print("[*] Loading %s word vectors..." % language)
    wvec, vocab = {}, {}
    embeddings = pickle.load(gzip.open(path+".gz", 'rb'))
    wvec[language] = embeddings[1]
    vocab[language] = list(embeddings[0])
    print ("\t%s" % (path)).ljust(60) + "OK"
    return wvec, vocab


if __name__ == "__main__":

    corpus_file = 'sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'
    # corpus_file = 'nda_proprietary_data2_sampled.jsonl'

    print('Loading corpus from', corpus_file)
    dataset = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')
    label_set = list(set(l for labels in dataset.y_train for l in labels))
    print('Label set size:', len(label_set))

    outpath = '/home/don/tools/gile/data'
    lang = 'english'

    _, vocab = load_word_vectors(lang, '/home/don/tools/gile/word_vectors/english.pkl')

    yh = get_label_counts(dataset.y_train + dataset.y_dev + dataset.y_test, lang)

    print("[*] Storing dev data...")
    X_dev, Y_dev = convert_data(dataset.x_dev[:100], dataset.y_dev[:100], vocab, label_set)
    if not os.path.exists(outpath+'/dev'):
        os.makedirs(outpath+'/dev')
    devfile = open(outpath+'/dev/%s.json' % lang, 'w')
    json.dump({'X_ids': X_dev, 'Y_ids': Y_dev, 'label_ids': yh.keys()}, devfile)

    print("[*] Storing test data...")
    X_test, Y_test = convert_data(dataset.x_test[:100], dataset.y_test[:100], vocab, label_set)
    if not os.path.exists(outpath+'/test'):
        os.makedirs(outpath+'/test')
    testfile = open(outpath+'/test/%s.json' % lang, 'w')
    json.dump({'X_ids': X_test, 'Y_ids': Y_test, 'label_ids': yh.keys()}, testfile)

    print("[*] Storing training data...")
    X, Y = convert_data(dataset.x_train[:100], dataset.y_train[:100], vocab, label_set)
    if not os.path.exists(outpath+'/train'):
        os.makedirs(outpath+'/train')
    trainfile = open(outpath+'/train/%s.json' % lang, 'w')
    json.dump({'X_ids': X, 'Y_ids': Y, 'label_ids': yh.keys()}, trainfile)
    print("[-] Finished.")
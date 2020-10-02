import numpy; numpy.random.seed(42)
import re
import pickle
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from utils import split_corpus, SplitDataSet, evaluate_multilabels, \
    tune_clf_thresholds


def train_classifiers(x_train: numpy.array, y_train: numpy.array) -> OneVsRestClassifier:
    clf = LogisticRegression(class_weight='balanced', max_iter=10000, solver='lbfgs')
    ovr = OneVsRestClassifier(clf, n_jobs=-1)
    ovr.fit(x_train, y_train)
    return ovr


def stringify_labels(y_vecs: numpy.array, mlb: MultiLabelBinarizer,
                     thresh: float = 0.5, label_threshs: Dict[str, float] = None) -> List[List[str]]:
    """
    Turn prediction probabilities into label strings
    :param y_vecs:
    :param mlb:
    :param thresh:
    :param label_threshs: Classification threshold per label
    :return:
    """
    y_pred: List[List[str]] = []
    if not label_threshs:
        label_threshs = {l: thresh for l in mlb.classes_}
    label_threshs = [label_threshs[l] for l in mlb.classes_]
    for prediction in y_vecs:
        label_indexes = numpy.where(prediction >= label_threshs)[0]
        if label_indexes.size > 0:  # One of the classes has triggered
            labels = set(numpy.take(mlb.classes_, label_indexes))
        else:
            labels = []
        y_pred.append(labels)
    return y_pred


def classify_by_labelname(x_test: List[str], y_train: List[List[str]],
                          prop_nda: bool = False) -> List[List[str]]:

    label_set = set(l for labels in y_train for l in labels)

    if prop_nda:  # Map proprietary label names to sec provision names
        import json
        prop2sec = json.load(open('../prop2sec_map.json'))
        sec2prop = {v: l for l, v in prop2sec.items()}
        label_set = {sec2prop[l].lower() for l in label_set if l in sec2prop}

    y_preds = []
    for i, x in enumerate(x_test):
        print(i, '\r', end='', flush=True)
        y_pred = []
        for orig_label in label_set:
            label = orig_label.replace('_', ' ')
            if re.search(r'\b%s\b' % label, x.lower()) \
                    or (label.endswith('s') and re.search(r'\b%s\b' % label[:-1], x.lower())):
                y_pred.append(orig_label)
        y_preds.append(y_pred)
    return y_preds


if __name__ == '__main__':

    predict_with_labelnames = False
    do_train = True
    do_test = True
    test_prop_nda = False

    import sys
    from pathlib import Path
    corpus_file = sys.argv[1]
    classifier_file = f'saved_models/logreg_{Path(corpus_file).stem}.pkl'
    Path(classifier_file).parent.mkdir(parents=True, exist_ok=True)

    print('Loading corpus from', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')
    label_set = set(l for labels in dataset.y_train for l in labels)
    print('Label set size:', len(label_set))

    if predict_with_labelnames:
        print('Predicting with label names')
        y_preds_labelnames = classify_by_labelname(dataset.x_test, dataset.y_train,
                                                   prop_nda=True)
        evaluate_multilabels(dataset.y_test, y_preds_labelnames, do_print=True)
        breakpoint()

    print('Vectorizing')
    tfidfizer = TfidfVectorizer(sublinear_tf=True)
    x_train_vecs = tfidfizer.fit_transform(dataset.x_train)
    x_test_vecs = tfidfizer.transform(dataset.x_test)
    x_dev_vecs = tfidfizer.transform(dataset.x_dev)

    mlb = MultiLabelBinarizer().fit(dataset.y_train)
    y_train_vecs = mlb.transform(dataset.y_train)
    y_test_vecs = mlb.transform(dataset.y_test)

    if do_train:
        print('Training LogReg')
        classifier = train_classifiers(x_train_vecs, y_train_vecs)
        with open(classifier_file, 'wb') as f:
            pickle.dump(classifier, f)
    else:
        print('Loading classifier')
        with open(classifier_file, 'rb') as f:
            classifier = pickle.load(f)

    if do_test:
        y_preds_lr_probs_dev = classifier.predict_proba(x_dev_vecs)
        label_threshs = tune_clf_thresholds(y_preds_lr_probs_dev, dataset.y_dev, mlb)
        y_preds_lr_probs = classifier.predict_proba(x_test_vecs)
        y_preds_lr = stringify_labels(y_preds_lr_probs, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_preds_lr, do_print=True)

    if test_prop_nda:
        nda_file = 'data/nda_proprietary_data2_sampled.jsonl'
        print('Loading corpus from', nda_file)

        dataset_nda: SplitDataSet = split_corpus(nda_file)
        
        nda_x_train_vecs = tfidfizer.transform(dataset_nda.x_train)
        nda_x_test_vecs = tfidfizer.transform(dataset_nda.x_test)
        nda_x_dev_vecs = tfidfizer.transform(dataset_nda.x_dev)
        nda_y_train = mlb.transform(dataset_nda.y_train)
        nda_y_test = mlb.transform(dataset_nda.y_test)
        nda_y_dev = mlb.transform(dataset_nda.y_dev)
    
        # Zero-shot; no training on prop data
        print('Zero-shot: train on LEDGAR, predict proprietary')
        y_preds_nda_probs_dev = classifier.predict_proba(nda_x_dev_vecs)
        label_threshs_nda = tune_clf_thresholds(y_preds_nda_probs_dev, dataset_nda.y_dev, mlb)
        y_preds_nda_probs = classifier.predict_proba(nda_x_test_vecs)
        y_preds_nda = stringify_labels(y_preds_nda_probs, mlb, label_threshs=label_threshs_nda)
        evaluate_multilabels(dataset_nda.y_test, y_preds_nda, do_print=True)

        print('In-domain: train on proprietary, predict proprietary')
        tfidfizer_prop = TfidfVectorizer(sublinear_tf=True)
        x_train_prop_vecs = tfidfizer_prop.fit_transform(dataset_nda.x_train)
        x_test_prop_vecs = tfidfizer_prop.transform(dataset_nda.x_test)
        x_dev_prop_vecs = tfidfizer_prop.transform(dataset_nda.x_dev)

        classifier_prop = train_classifiers(x_train_prop_vecs, nda_y_train)
        y_preds_prop_prob_dev = classifier_prop.predict_proba(x_dev_prop_vecs)
        label_threshs_prop = tune_clf_thresholds(y_preds_prop_prob_dev, dataset_nda.y_dev, mlb)
        y_preds_prop_prob_test = classifier_prop.predict_proba(x_test_prop_vecs)
        y_preds_prop = stringify_labels(y_preds_prop_prob_test, mlb, label_threshs=label_threshs_prop)
        evaluate_multilabels(dataset_nda.y_test, y_preds_prop, do_print=True)

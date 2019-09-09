import pdb
import numpy; numpy.random.seed(42)
import re
from typing import List, Set, DefaultDict, Dict, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression



def train_classifiers(train_x: List[str], train_y: List[Set[str]], tf_idf: TfidfVectorizer, mlb: MultiLabelBinarizer) \
        -> OneVsRestClassifier:
    """
    Train a one-vs-rest classifier per label
    :param train_x: List of texts
    :param train_y: List of labels
    :param tf_idf: Fitted TF IDF transformer
    :param mlb: Fitted MultiLabelBinarizer
    :return: Trained classifier
    """
    clf = LogisticRegression(class_weight='balanced', max_iter=10000, solver='lbfgs')
    ovr = OneVsRestClassifier(clf, n_jobs=-1)
    y = mlb.transform(train_y)
    x = tf_idf.transform(train_x)
    ovr.fit(x, y)
    return ovr


def stringify_labels(y_pred_bin:numpy.array, mlb:MultiLabelBinarizer,
                     thresh: float = 0.5, label_threshs: Dict[str, float] = None) -> List[List[str]]:
    """
    Turn prediction probabilities into label strings
    :param y_pred_bin:
    :param mlb:
    :param thresh:
    :param label_threshs: Classification threshold per label
    :return:
    """
    y_pred: List[List[str]] = []
    if not label_threshs:
        label_threshs = {l: thresh for l in mlb.classes_}
    label_threshs = [label_threshs[l] for l in mlb.classes_]
    for prediction in y_pred_bin:
        label_indexes = numpy.where(prediction >= label_threshs)[0]
        if label_indexes.size > 0:  # One of the classes has triggered
            labels = set(numpy.take(mlb.classes_, label_indexes))
        else:
            labels = []
        y_pred.append(labels)
    return y_pred


def evaluate_multilabels(y: List[List[str]], y_preds: List[List[str]], do_print: bool = False) -> DefaultDict[str, Dict[str, float]]:
    """
    Print classification report with multilabels
    :param y: Gold labels
    :param y_preds: Predicted labels
    :param do_print: Whether to print results
    :return: Dict of scores per label and overall
    """
    # Label -> TP/FP/FN -> Count
    label_eval: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    assert len(y) == len(y_preds), "List of predicted and gold labels are of unequal length"
    for y_true, y_pred in zip(y, y_preds):
        for label in y_true:
            if label in y_pred:
                label_eval[label]['tp'] += 1
            else:
                label_eval[label]['fn'] += 1
        for label in y_pred:
            if label not in y_true:
                label_eval[label]['fp'] += 1

    max_len = max([len(l) for l in label_eval.keys()])
    if do_print:
        print('\t'.join(['Label'.rjust(max_len, ' '), 'Prec'.ljust(4, ' '), 'Rec'.ljust(4, ' '), 'F1'.ljust(4, ' '),
                     'Support']))

    eval_results: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_f1, all_rec, all_prec = [], [], []
    for label in sorted(label_eval.keys()):
        cnts = label_eval[label]
        if not cnts['tp'] == 0:
            prec = cnts['tp'] / (cnts['tp'] + cnts['fp'])
            rec = cnts['tp'] / (cnts['tp'] + cnts['fn'])
            f1 = 2 * prec * rec / (prec + rec)
        else:
            prec, rec, f1 = 0.00, 0.00, 0.00
        eval_results[label]['prec'] = prec
        eval_results[label]['rec'] = rec
        eval_results[label]['f1'] = f1
        eval_results[label]['support'] = cnts['tp'] + cnts['fn']
        all_f1.append(f1)
        all_rec.append(rec)
        all_prec.append(prec)
        if do_print:
            print('\t'.join([label.rjust(max_len, ' '),
                         ('%.2f' % round(prec, 2)).ljust(4, ' '),
                         ('%.2f' % round(rec, 2)).ljust(4, ' '),
                         ('%.2f' % round(f1, 2)).ljust(4, ' '),
                         str(cnts['tp'] + cnts['fn']).rjust(5, ' ')]))

    eval_results['Macro']['prec'] = sum(all_prec) / len(all_prec)
    eval_results['Macro']['rec'] = sum(all_rec) / len(all_rec)
    eval_results['Macro']['f1'] = sum(all_f1) / len(all_f1)
    eval_results['Macro']['support'] = len(y)

    # Micro
    all_tp = sum(label_eval[label]['tp'] for label in label_eval)
    all_fp = sum(label_eval[label]['fp'] for label in label_eval)
    all_fn = sum(label_eval[label]['fn'] for label in label_eval)
    eval_results['Micro']['prec'] = all_tp / (all_tp + all_fp)
    eval_results['Micro']['rec'] = all_tp / (all_tp + all_fn)
    micro_prec = eval_results['Micro']['prec']
    micro_rec = eval_results['Micro']['rec']
    eval_results['Micro']['f1'] = 2 * (micro_rec * micro_prec) / (micro_rec + micro_prec)
    eval_results['Micro']['support'] = len(y)

    if do_print:
        print('Macro Avg. Rec:', round(eval_results['Macro']['rec'], 2))
        print('Macro Avg. Prec:', round(eval_results['Macro']['prec'], 2))
        print('Macro F1:', round(eval_results['Macro']['f1'], 2))
        print()
        print('Micro Avg. Rec:', round(eval_results['Micro']['rec'], 2))
        print('Micro Avg. Prec:',  round(eval_results['Micro']['prec'], 2))
        print('Micro F1:', round(eval_results['Micro']['f1'], 2))

    return eval_results


def classify_by_labelname(x_train, x_test, y_train, y_test):
    label_set = set(l for labels in y_train for l in labels)
    y_preds = []
    for x in x_test:
        y_pred = []
        for label in label_set:
            if label in x.lower() or (label.endswith('s') and label[:-1] in x.lower()):
                y_pred.append(label)
        y_preds.append(y_pred)
    return y_preds


def tune_clf_thresholds(test_x, test_y, classifier, mlb):
    y_pred_bin: numpy.array = classifier.predict_proba(test_x)
    thresh_range = [t / 10.0 for t in range(1, 10)]
    all_results = dict()
    print('Evaluating classification threholds')
    for t in thresh_range:
        print(t)
        y_pred = stringify_labels(y_pred_bin, mlb, thresh=t)
        eval_results = evaluate_multilabels(test_y, y_pred, do_print=False)
        all_results[t] = eval_results

    label_threshs: Dict[str, float] = dict()
    for label in mlb.classes_:
        best_thresh, best_f1 = 0.1, 0.0
        for curr_thresh in thresh_range:
            curr_f1 = all_results[curr_thresh][label]['f1']
            if curr_f1 > best_f1:
                best_thresh = curr_thresh
                best_f1 = curr_f1
        label_threshs[label] = best_thresh

    y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
    return y_pred, label_threshs


if __name__ == '__main__':
    import json

    # TODO try classifying sparse labels! compare results to frequent labels etc.
    corpus_file = 'sec_corpus_2016-2019_clean_sampled.jsonl'



    print('Predicting with label names')
    y_preds_labelnames = classify_by_labelname(x_train, x_test, y_train, y_test)
    evaluate_multilabels(y_test, y_preds_labelnames, do_print=True)

    mlb = MultiLabelBinarizer().fit(y)
    print('Vectorizing')
    tfidf = TfidfVectorizer(sublinear_tf=True).fit(x_train)
    print('Training LogReg')
    classifier = train_classifiers(x_train, y_train, tfidf, mlb)

    x_test = tfidf.transform(x_test)
    y_preds_lr, _ = tune_clf_thresholds(x_test, y_test, classifier, mlb)
    # y_preds_bin = classifier.predict_proba(x_test)
    # y_preds_lr = stringify_labels(y_preds_bin, mlb)
    evaluate_multilabels(y_test, y_preds_lr, do_print=True)


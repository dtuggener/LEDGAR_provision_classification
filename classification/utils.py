import json
import numpy
import re
from typing import List, Union, Dict, DefaultDict, Tuple
from collections import defaultdict
# from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


"""
@dataclass
class SplitDataSet:
    x_train: List[str]
    y_train: List[List[str]]
    x_test: List[str]
    y_test: List[List[str]]
    x_dev: Union[List[str], None]
    y_dev: Union[List[List[str]], None]
"""

# python3.6
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


def split_corpus(corpus_file: str, use_dev: bool = True,
                 test_size: float = 0.2, dev_size: Union[float, None] = 0.1,
                 random_state: int = 42) -> SplitDataSet:
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
    for line in open(corpus_file, encoding='utf-8'):
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


def evaluate_multilabels(y: List[List[str]], y_preds: List[List[str]],
                         do_print: bool = False) -> DefaultDict[str, Dict[str, float]]:
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
            f1 = (2 * prec * rec) / (prec + rec)
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
                         str(cnts['tp'] + cnts['fn']).rjust(5, ' ')
                         ]))

    eval_results['Macro']['prec'] = sum(all_prec) / len(all_prec)
    eval_results['Macro']['rec'] = sum(all_rec) / len(all_rec)
    if eval_results['Macro']['prec'] + eval_results['Macro']['rec'] == 0:
        eval_results['Macro']['f1'] = 0.0
    else:
        eval_results['Macro']['f1'] = (2 * eval_results['Macro']['prec'] * eval_results['Macro']['rec']) / \
                                  (eval_results['Macro']['prec'] + eval_results['Macro']['rec'])
    eval_results['Macro']['support'] = len(y)

    # Micro
    all_tp = sum(label_eval[label]['tp'] for label in label_eval)
    all_fp = sum(label_eval[label]['fp'] for label in label_eval)
    all_fn = sum(label_eval[label]['fn'] for label in label_eval)
    if all_fp == 0:
        eval_results['Micro']['prec'] = 0
        eval_results['Micro']['rec'] = 0
        eval_results['Micro']['f1'] = 0
    else:
        eval_results['Micro']['prec'] = all_tp / (all_tp + all_fp)
        eval_results['Micro']['rec'] = all_tp / (all_tp + all_fn)
        micro_prec = eval_results['Micro']['prec']
        micro_rec = eval_results['Micro']['rec']
        if micro_prec + micro_rec == 0:
            eval_results['Micro']['f1'] = 0.0
        else:
            eval_results['Micro']['f1'] = (2 * micro_rec * micro_prec) / (micro_rec + micro_prec)
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


def average_embeddings(x_embedded: List[List[float]], avg_method: str, embedding_dims: int) -> numpy.array:
    """
    Average word embeddings in a sentence to get a sentence represenation
    :param x_embedded: List of embeddings
    :param avg_method: Method to average the embeddings
    :param embedding_dims: Dimensionality of the word embeddings
    :return: Averaged word embeddings
    """
    if x_embedded:
        if avg_method == 'mean':
            return numpy.mean(x_embedded, axis=0)
        elif avg_method == 'sum':
            return numpy.sum(x_embedded, axis=0)
        elif avg_method == 'max_pool':
            return numpy.max(x_embedded, axis=0)
    else:
        return numpy.zeros(embedding_dims)


def embed(x_s: List[str], embeddings: numpy.array, vocab: Dict, tfidfizer: TfidfVectorizer = None,
          avg_method: str = 'mean', use_tfidf: bool = True) -> numpy.array:
    """
    Transform texts to averaged word embeddings, weighted by TF IDF
    :param x_s: List of texts
    :param embeddings: Embedding matrix
    :param vocab: Vocabulary covered by embeddings
    :param avg_method: how to average embeddings; 'mean', 'sum', or 'max_pool'
    :param use_tfidf: whether to weigh each embedding by it's tf idf weight
    :return: List of averaged word embeddings
    """
    x_vecs = []
    embedding_dims = embeddings.shape[1]
    if use_tfidf:
        if not tfidfizer:
            tfidfizer = TfidfVectorizer(norm=None).fit(x_s)
        ix2word = {v: k for k, v in tfidfizer.vocabulary_.items()}
        x_tfidf = tfidfizer.transform(x_s).toarray()
        for x in x_tfidf:
            x_embedded = []
            non_zeros = numpy.where(x > 0)[0]  # Find non-zero entries
            for ix in non_zeros:
                word = ix2word[ix]
                if word in vocab:
                    x_embedded.append(x[ix] * embeddings[vocab[word]])  # Multiply the embedding with the TF IDF weight
            x_avg = average_embeddings(x_embedded, avg_method, embedding_dims)
            x_vecs.append(x_avg)
    else:
        for x in x_s:
            x_embedded = []
            words = re.findall('\w+', x.lower())
            for word in words:
                if word in vocab:
                    x_embedded.append(embeddings[vocab[word]])
            x_avg = average_embeddings(x_embedded, avg_method, embedding_dims)
            x_vecs.append(x_avg)

    return numpy.array(x_vecs)


def tune_clf_thresholds(y_pred_vecs: numpy.array, test_y: List[List[str]],
                        mlb: MultiLabelBinarizer,
                        objective: str = 'f1',
                        min_freq: int = 5) -> Dict[str, float]:

    assert objective in {'f1', 'balanced', 'std'}, \
        f'{objective} not a valid tuning objective for classifier threshold'

    thresh_range = [t / 100.0 for t in range(1, 100)]
    all_results = dict()
    for thresh in thresh_range:
        y_pred = stringify_labels(y_pred_vecs, mlb, thresh=thresh)
        eval_results = evaluate_multilabels(test_y, y_pred, do_print=False)
        all_results[thresh] = eval_results

    # Don't tune threshholds for labels with less than min_freq support in dev
    eval_results = all_results[thresh_range[0]]
    low_freq_labels = {l for l, res_dict in eval_results.items() if
                       res_dict['support'] < min_freq}

    label_threshs: Dict[str, float] = dict()
    if objective == 'f1':
        for label in mlb.classes_:
            if label in low_freq_labels:
                label_threshs[label] = 0.5
            else:
                best_thresh, best_f1 = min(thresh_range), 0.0
                for t in thresh_range:
                    if all_results[t][label]['f1'] >= best_f1:  # Changing this to '>' favors recall more.
                        best_thresh, best_f1 = t, all_results[t][label]['f1']
                label_threshs[label] = best_thresh

    elif objective == 'balanced':
        for label in mlb.classes_:
            if label in low_freq_labels:
                label_threshs[label] = 0.5
            else:
                best_thresh, best_balance = min(thresh_range), 0.0
                for t in thresh_range:
                    balance = all_results[t][label]['rec'] \
                              + all_results[t][label]['prec'] \
                              + all_results[t][label]['f1']
                    if balance >= best_balance:
                        best_thresh, best_balance = t, balance
                label_threshs[label] = best_thresh

    elif objective == 'std':
        for label in mlb.classes_:
            if label in low_freq_labels:
                label_threshs[label] = 0.5
            else:
                best_thresh, best_balance = min(thresh_range), 1.0
                for t in thresh_range:
                    balance = numpy.std([all_results[t][label]['rec'],
                                         all_results[t][label]['prec'],
                                         all_results[t][label]['f1']])
                    if balance <= best_balance:
                        best_thresh, best_balance = t, balance
                label_threshs[label] = best_thresh

    return label_threshs


def calc_class_weights(y: numpy.array, label2ix: Dict[str, int])\
        -> Dict[str, float]:
    total = 0
    class_weights = numpy.zeros(len(label2ix), dtype=numpy.float32)
    for labels in y:
        class_weights += labels
        total += 1
    class_weights = total / (len(label2ix) * class_weights)
    return class_weights


if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean_sampled.jsonl'
    ds = split_corpus(corpus_file)
    breakpoint()



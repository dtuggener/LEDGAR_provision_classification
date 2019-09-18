import json
from typing import List, Union, Dict, DefaultDict, Tuple
from collections import defaultdict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class SplitDataSet:
    x_train: List[str]
    y_train: List[List[str]]
    x_test: List[str]
    y_test: List[List[str]]
    x_dev: Union[List[str], None]
    y_dev: Union[List[List[str]], None]


def split_corpus(corpus_file: str, use_dev: bool = True,
                 test_size: float = 0.2, dev_size: Union[float, None] = 0.1,
                 random_state: int = 42) -> SplitDataSet:
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
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


def tuple_contains(tup1: Tuple, tup2: Tuple) -> Tuple[bool, int]:
    """Check whether tuple 1 contains tuple 2"""
    len_tup1, len_tup2 = len(tup1), len(tup2)
    for i in range(0, len_tup1 + 1 - len_tup2):
        if tup1[i:i + len_tup2] == tup2:
            return True, i
    return False, -1


if __name__ == '__main__':
    corpus_file = 'sec_corpus_2016-2019_clean_sampled.jsonl'
    ds = split_corpus(corpus_file)
    breakpoint()



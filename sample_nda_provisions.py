"""
From the labels provisions, sample those that have been deemed relevant for NDAs
"""

import re
import json
from typing import Dict, List
from labelset_hierarchy import get_base_forms


def read_mapping(mapping_tsv: str = 'label_mapping.tsv') -> Dict[str, str]:
    label_map: Dict[str, str] = dict()
    label_delimiters = re.compile('[/;,]')
    for line in open(mapping_tsv):
        line = line.split('\t')
        if len(line) == 2:
            source_label, mapped_labels = line
            mapped_labels = label_delimiters.split(mapped_labels.strip())
            for mapped_label in mapped_labels:
                label_map[mapped_label.strip().lower()] = source_label.strip().lower()
    return label_map


def sample_provisions(x, y, doc_ids, labels, base_forms):
    sampled_x, sampled_y, sampled_doc_ids = [], [], []
    for x_, y_, doc_id in zip(x, y, doc_ids):
        sampled_labels = []
        for label in y_:
            label_baseform = ' '.join([base_forms.get(lt, lt) for lt in label.split(' ')])
            if label_baseform in labels:
                sampled_labels.append(label_baseform)
        if sampled_labels:
            sampled_x.append(x_)
            sampled_y.append(sampled_labels)
            sampled_doc_ids.append(doc_id)
    return sampled_x, sampled_y, sampled_doc_ids


if __name__ == '__main__':

    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'
    print('Loading data from', corpus_file)
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
    for line in open(corpus_file):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    label_set = set([l for labels in y for l in labels])

    base_forms = get_base_forms(label_set)
    label_map = read_mapping()
    label_map_baseform = {' '.join([base_forms.get(lt, lt) for lt in l.split(' ')]): v for l, v in label_map.items()}

    print('Sampling')
    sampled_x, sampled_y, sampled_doc_ids = sample_provisions(x, y, doc_ids, label_map_baseform, base_forms)
    label_set_sampled = set([l for labels in sampled_y for l in labels])
    print('Found {} of {} labels'.format(len(label_set_sampled), len(label_map_baseform)))

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_NDA_PTs.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(x, y, doc_ids):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')

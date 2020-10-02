"""
From the labels provisions, sample those that have been deemed relevant for NDAs
"""

import re
import json
from typing import Dict, List
from labelset_hierarchy import get_base_forms


def read_mapping(mapping_tsv: str = 'label_mapping.tsv') -> Dict[str, str]:
    prop_map = json.load(open('prop2sec_map.json', 'r', encoding='utf8'))
    label_map: Dict[str, str] = dict()
    label_delimiters = re.compile('[/;,]')
    for line in open(mapping_tsv):
        # TSV is prop_label -> sec_label
        line = line.split('\t')
        if len(line) == 2:
            source_label, mapped_labels = line
            mapped_labels = label_delimiters.split(mapped_labels.strip())
            for mapped_label in mapped_labels:
                label_map[mapped_label.strip().lower()] = prop_map[source_label].strip()
    return label_map


def sample_provisions(x, y, doc_ids, labels, base_forms):
    sampled_x, sampled_y, sampled_doc_ids = [], [], []
    for x_, y_, doc_id in zip(x, y, doc_ids):
        sampled_labels = []
        for label in y_:
            label_baseform = ' '.join([base_forms.get(lt, lt) for lt in label.split(' ')])
            if label_baseform.lower() in labels:
                sampled_labels.append(labels[label_baseform.lower()])
        if sampled_labels:
            sampled_x.append(x_)
            sampled_y.append(sampled_labels)
            sampled_doc_ids.append(doc_id)
    return sampled_x, sampled_y, sampled_doc_ids


if __name__ == '__main__':

    # Maps SEC labels -> proprietary labels
    label_map = read_mapping()

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
    label_map_baseform = {' '.join([base_forms.get(lt, lt) for lt in l.split(' ')]): v for l, v in label_map.items()}

    print('Sampling')
    # Returns proprietary labels
    sampled_x, sampled_y, sampled_doc_ids = sample_provisions(x, y, doc_ids, label_map_baseform, base_forms)
    label_set_sampled = set([l for labels in sampled_y for l in labels])
    print('Found {} of {} labels'.format(len(label_set_sampled), len(set(label_map_baseform.values()))))

    # Read proprietary data
    prop_data_file = 'nda_proprietary_data2.jsonl'
    print('Loading data from', prop_data_file)

    x_prop: List[str] = []
    y_prop: List[List[str]] = []
    doc_ids_prop: List[str] = []
    for line in open(prop_data_file):
        labeled_provision = json.loads(line)
        sampled_labels = []
        for label in labeled_provision['label']:
            if label in label_set_sampled:
                sampled_labels.append(label)
        if sampled_labels:
            x_prop.append(labeled_provision['provision'])
            y_prop.append(sampled_labels)
            doc_ids_prop.append(labeled_provision['source'])

    label_set_prop = set(l for labels in y_prop for l in labels)
    print('Sampled', len(label_set_prop), 'labels')

    # Only take those PTs from LEDGAR that are actually annotated in proprietary data
    new_map = {l: v for l, v in label_map_baseform.items() if v in label_set_prop}
    sampled_x, sampled_y, sampled_doc_ids = sample_provisions(x, y, doc_ids, new_map, base_forms)

    print('Writing output')
    with open(corpus_file.replace('.jsonl', '_NDA_PTs2.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(sampled_x, sampled_y, sampled_doc_ids):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')

    with open(prop_data_file.replace('.jsonl', '_sampled.jsonl'), 'w',  encoding='utf8') as f:
        for provision, labels, doc_id in zip(x_prop, y_prop, doc_ids_prop):
            json.dump({"provision": provision, "label": labels, "source": doc_id}, f, ensure_ascii=False)
            f.write('\n')
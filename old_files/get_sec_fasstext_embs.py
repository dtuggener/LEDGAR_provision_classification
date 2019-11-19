"""
Get words from sec corpus and respective fasttext embeddings
"""

import re
import fasttext
from classification.utils import SplitDataSet, split_corpus

re_read = False

if re_read:
    corpus_file = 'sec_corpus_2016-2019_clean.jsonl'

    print('Loading corpus', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')

    vocab = set()
    print('reading')
    for x in dataset.x_train:
        words = re.findall('\w+', x.lower())
        vocab.update(words)

    print('writing')
    with open('sec_vocab.txt', 'w') as f:
        for word in vocab:
            f.write(word + '\n')

print('Loading embeddings')
embedding_file = 'classification/data/cc.en.300.bin'
embeddings = fasttext.load_model(embedding_file)

print('writing vecs')
with open('sec_fasttext_vecs.txt', 'w') as f:
    words = open('sec_vocab.txt').readlines()
    f.write(str(len(words)) + ' ' + '300\n')
    for word in words:
        vec = embeddings.get_word_vector(word)
        f.write(word.strip() + ' ' + ' '.join([str(v) for v in vec]) + '\n')


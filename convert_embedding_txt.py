'''
Convert standard word embedding file to a .json containing word -> id mapping
and a .npy numpy file that contains the embeddings
filters out multi word forms
adds tokens 'UNK' and 'PAD'
'''

import sys

import json

import numpy as np

if __name__ == '__main__':
    embedding_file = sys.argv[1]
    try:
        out_name = sys.argv[2]
    except IndexError:
        out_name = sys.argv[1]

    with open(embedding_file, 'r') as fin:
        first = True
        current_ix = 0
        for line in fin:
            tokens = line.strip().split(' ')
            if first:
                n_vocab = int(tokens[0])
                n_dim = int(tokens[1])
                #embeddings = np.zeros((0, n_dim), dtype=np.float32)
                embeddings = []
                vocab = {}
                first = False
            else:
                words = tokens[:-n_dim]
                vals = tokens[-n_dim:]
                token = ' '.join(words)
                if not token in vocab:
                    embedded = [float(x) for x in vals]
                    #for i, val in enumerate(vals):
                    #    embedded[i] = float(val)
                    #embeddings = np.append(embeddings, [embedded], axis=0)
                    embeddings.append(embedded)
                    vocab[token] = current_ix
                    current_ix += 1
                else:
                    print(token)

    embeddings = np.array(embeddings, dtype=np.float32)

    vocab['UNK'] = current_ix
    #embeddings[current_ix] = np.mean(embeddings[:current_ix, :], axis=0)
    embeddings = np.append(embeddings, [np.mean(embeddings[:current_ix, :], axis=0)], axis=0)
    current_ix += 1

    vocab['PAD'] = current_ix
    #embeddings[current_ix] = np.zeros(n_dim)
    embeddings = np.append(embeddings, [np.zeros(n_dim)], axis=0)
    current_ix += 1

    with open(f'{out_name}_vocab.json', 'w') as fout:
        json.dump(vocab, fout)

    np.save(
        f'{out_name}_data.npy',
        embeddings,
        fix_imports=True,
        allow_pickle=False)

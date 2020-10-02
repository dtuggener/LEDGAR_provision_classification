"""
Sentence classification with self-attention in keras
"""

import json
import numpy; numpy.random.seed(42)
import re
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from attn_layer import AttentionLayer
from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds, calc_class_weights


def build_model(max_sent_length, vocab2int, embeddings, num_labels):
    embedding_dim = embeddings[0].shape[0]

    input_layer = Input(shape=(max_sent_length,))

    embedding_layer = Embedding(len(vocab2int),
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sent_length,
                                trainable=False,
                                mask_zero=True)

    embedded = embedding_layer(input_layer)

    l_att = AttentionLayer()(embedded)

    dense_1 = Dense(embedding_dim * 2,
                    activation='relu')(l_att)

    classifier = Dense(num_labels,
                       activation='sigmoid')(dense_1)

    model = Model(inputs=input_layer,
                  outputs=classifier)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def tune_threshs(probas, truth):
    res = numpy.zeros(probas.shape[1])

    for i in range(probas.shape[1]):
        thresh = max(
            numpy.linspace(
                numpy.min(probas[:, i]),
                numpy.max(probas[:, i]),
                num=100,
            ),
            key=lambda t: f1_score(y_true=truth[:, i], y_pred=(probas[:, i] > t))
        )
        res[i] = thresh

    res[res == 0] = 0.5

    return res


def count_oovs(x):
    oovs = Counter()
    for x_ in x:
        oov = [w for w in re.findall('\w+', x_) if w.lower() not in vocab]
        for w in oov:
            oovs[w] += 1
    return oovs


if __name__ == '__main__':

    do_train = True
    do_test = True
    do_test_nda = False
    classification_thresh = 0.5

    import sys
    from pathlib import Path
    corpus_file = sys.argv[1]
    model_name = f'saved_models/MLP_avg_{Path(corpus_file).stem}.h5'
    Path(model_name).parent.mkdir(parents=True, exist_ok=True)

    embedding_file = sys.argv[2]
    vocab_file = sys.argv[3]

    embeddings = numpy.load(embedding_file)
    vocab = json.load(open(vocab_file))
    int2vocab = {i: w for w, i in vocab.items()}
    embedding_dim = embeddings[0].shape[0]

    print('Loading corpus', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')

    # oov_counts = count_oovs(dataset.x_train)
    # breakpoint()

    mlb = MultiLabelBinarizer().fit(dataset.y_train)
    num_classes = mlb.classes_.shape[0]
    train_y = mlb.transform(dataset.y_train)
    test_y = mlb.transform(dataset.y_test)
    dev_y = mlb.transform(dataset.y_dev)

    train_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                   for x_ in dataset.x_train]
    test_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                  for x_ in dataset.x_test]
    dev_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                 for x_ in dataset.x_dev]

    max_sent_length = max([len(x_) for x_ in train_x_int])
    train_x = pad_sequences(train_x_int, max_sent_length)
    test_x = pad_sequences(test_x_int, max_sent_length)
    dev_x = pad_sequences(dev_x_int, max_sent_length)

    if do_train:
        model = build_model(max_sent_length, vocab, embeddings, num_classes)
        print(model.summary())

        class_weights = calc_class_weights(train_y, mlb.classes_)
        class_weights = {i: w for i, w in enumerate(class_weights)}

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       restore_best_weights=True)

        try:
            model.fit(train_x, train_y,
                      batch_size=32,
                      epochs=50,
                      verbose=1,
                      validation_data=(dev_x, dev_y),
                      class_weight=class_weights,
                      callbacks=[early_stopping]
                      )
        except KeyboardInterrupt:
            pass

        model.save(model_name, overwrite=True)

    else:
        model = load_model(model_name,
                           custom_objects={'AttentionLayer': AttentionLayer})

    if do_test:
        print('predicting')
        y_pred_bin_dev = model.predict(dev_x, verbose=1)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)
        y_pred_bin = model.predict(test_x, verbose=1)
        y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

    if do_test_nda:
        nda_file = 'data/nda_proprietary_data2_sampled.jsonl'
        print('Loading corpus from', nda_file)

        dataset: SplitDataSet = split_corpus(nda_file)
        train_y = mlb.transform(dataset.y_train)
        test_y = mlb.transform(dataset.y_test)
        dev_y = mlb.transform(dataset.y_dev)

        test_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                      for x_ in dataset.x_test]
        dev_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                     for x_ in dataset.x_dev]

        test_x = pad_sequences(test_x_int, max_sent_length)
        dev_x = pad_sequences(dev_x_int, max_sent_length)

        print('predicting NDA')
        y_pred_bin_dev = model.predict(dev_x, verbose=1)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)
        y_pred_bin = model.predict(test_x, verbose=1)
        y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

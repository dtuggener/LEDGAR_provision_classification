"""
Sentence classification with self-attention in keras
"""

import json
import numpy; numpy.random.seed(42)
import re
from typing import List
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

import keras.backend as K
from keras.layers import Input, Embedding, TimeDistributed, Dense, \
    GlobalAveragePooling1D, Multiply, Lambda, LSTM
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
from keras.callbacks import EarlyStopping, TensorBoard
from keras import initializers

from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds

from attn_layer import AttentionLayer


def build_model(max_sent_length, vocab2int, embeddings, num_labels):
    # Model with self-attention
    embedding_dim = embeddings[0].shape[0]
    input_layer = Input(shape=(max_sent_length,))
    embedding_layer = Embedding(len(vocab2int), embedding_dim,
                                weights=[embeddings], input_length=max_sent_length,
                                trainable=False, mask_zero=True)
    embedded = embedding_layer(input_layer)
    # lstm = LSTM(64, return_sequences=True)(embedded)
    l_att = AttentionLayer()(embedded)
    classifier = Dense(num_labels, activation='sigmoid')(l_att)
    model = Model(inputs=input_layer, outputs=classifier)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    do_train = True
    do_test = True
    classification_thresh = 0.5

    # corpus_file = 'data/sec_corpus_2016-2019_clean_proto.jsonl'
    # model_name = 'MLP_attn_proto.h5'

    corpus_file = 'data/sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'
    model_name = 'MLP_attn_leaves_subsampled.h5'

    embedding_file = 'data/wiki.multi.en.vec_data.npy'
    vocab_file = 'data/wiki.multi.en.vec_vocab.json'
    embeddings = numpy.load(embedding_file)
    vocab = json.load(open(vocab_file))
    int2vocab = {i: w for  w, i in vocab.items()}
    embedding_dim = embeddings[0].shape[0]

    print('Loading corpus', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')

    mlb = MultiLabelBinarizer().fit(dataset.y_train)
    num_classes = mlb.classes_.shape[0]
    train_y = mlb.transform(dataset.y_train)
    test_y = mlb.transform(dataset.y_test)
    dev_y = mlb.transform(dataset.y_dev)

    train_x_int = [[vocab[w] for w in re.findall('\w+', x_) if w in vocab] for x_ in dataset.x_train]
    test_x_int = [[vocab[w] for w in re.findall('\w+', x_) if w in vocab] for x_ in dataset.x_test]
    dev_x_int = [[vocab[w] for w in re.findall('\w+', x_) if w in vocab] for x_ in dataset.x_dev]

    max_sent_length = max([len(x_) for x_ in train_x_int])
    train_x = pad_sequences(train_x_int, max_sent_length)
    test_x = pad_sequences(test_x_int, max_sent_length)
    dev_x = pad_sequences(dev_x_int, max_sent_length)

    # Remove zero-valued training examples and labels; they break fit()!
    zero_ixs = [i for i, x_ in enumerate(train_x_int) if not  x_]
    train_x = numpy.delete(train_x, zero_ixs, axis=0)
    train_x_int = numpy.delete(train_x_int, zero_ixs, axis=0)
    train_x_str = numpy.delete(dataset.x_test, zero_ixs, axis=0)
    train_y = numpy.delete(train_y, zero_ixs, axis=0)
    train_y_str = numpy.delete(dataset.y_train, zero_ixs, axis=0)

    zero_ixs = [i for i, x_ in enumerate(dev_x_int) if not x_]
    dev_x = numpy.delete(dev_x, zero_ixs, axis=0)
    dev_x_int = numpy.delete(dev_x_int, zero_ixs, axis=0)
    dev_x_str = numpy.delete(dataset.x_dev, zero_ixs, axis=0)
    dev_y = numpy.delete(dev_y, zero_ixs, axis=0)
    dev_y_str = numpy.delete(dataset.y_dev, zero_ixs, axis=0)

    if do_train:
        model = build_model(max_sent_length, vocab, embeddings, num_classes)
        print(model.summary())

        # Calculate class weights
        all_labels: List[str] = [l for labels in train_y_str for l in labels]
        label_counts = Counter(all_labels)
        sum_labels_counts = sum(label_counts.values())
        # TODO calculate weights in sklearn fashion
        class_weight = {numpy.where(mlb.classes_ == label)[0][0]: 1 - (cnt / sum_labels_counts) for label, cnt in
                        label_counts.items()}

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3,
                                       restore_best_weights=True)
        tensor_board = TensorBoard()
        model.fit(train_x, train_y, batch_size=32, epochs=50,
                  verbose=1,
                  validation_data=(dev_x, dev_y),
                  # validation_split=0.1,
                  # class_weight=class_weight,
                  #callbacks=[early_stopping, tensor_board])
                  callbacks=[early_stopping])

        model.save('saved_models/%s' % model_name, overwrite=True)
    else:
        model = load_model('saved_models/%s' % model_name)

    # plot_model(model, to_file='/tmp/%s.png' % model_name)

    if do_test:
        print('predicting')
        y_pred_bin = model.predict(test_x, verbose=1)
        # TODO add classifier tuning
        y_pred = stringify_labels(y_pred_bin, mlb, thresh=classification_thresh)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

"""
Sentence classification with self-attention in keras
"""

import json
import numpy; numpy.random.seed(42)
import re
from typing import List
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras.layers import Input, Embedding, TimeDistributed, Dense, \
    GlobalAveragePooling1D, Multiply, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds


def my_repeat_vecs(x, rep, axis):
    from tensorflow.keras.backend import repeat_elements
    return repeat_elements(x, rep, axis)


def build_model(max_sent_length, vocab2int, embeddings, num_labels):
    # Model with self-attention
    embedding_dim = embeddings[0].shape[0]
    input_layer = Input(shape=(max_sent_length,))
    embedding_layer = Embedding(len(vocab2int), embedding_dim,
                                weights=[embeddings], input_length=max_sent_length,
                                trainable=False, mask_zero=True)
    embedded = embedding_layer(input_layer)
    self_attn = TimeDistributed(Dense(1))(embedded)

    # Repeat the self_attn output so we can multiply it with the word embeddings
    repeated_self_attn = Lambda(my_repeat_vecs, arguments={'rep': embedding_dim, 'axis': -1})(self_attn)

    sent_weighted = Multiply()([repeated_self_attn, embedded])
    sent_averaged = GlobalAveragePooling1D()(sent_weighted)
    #hidden_dense = Dense(512, activation='relu')(sent_averaged)
    classifier = Dense(num_labels, activation='sigmoid')(sent_averaged)
    model = Model(inputs=input_layer, outputs=classifier)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_weights(att_weights, padded_ints, int2vocab):
    words_weights = []
    for weight, word_ix in zip(att_weights.flatten(), padded_ints.flatten()):
        if not word_ix == 0:
            words_weights.append((int2vocab[word_ix], weight))
    return words_weights


def get_rgb(weight):
    aR, aG, aB = 0, 0, 255
    bR, bG, bB = 255, 0, 0
    red = (bR - aR) * weight + aR
    green = (bG - aG) * weight + aG
    blue = (bB - aB) * weight + aB
    return int(round(red)), int(round(green)), int(round(blue))


if __name__ == '__main__':

    do_train = True
    do_test = True
    classification_thresh = 0.5

    corpus_file = 'data/sec_corpus_2016-2019_clean_proto.jsonl'
    model_name = 'MLP_attn_proto.h5'

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
    zero_ixs = [i for i, x_ in enumerate(train_x) if sum(x_) == 0]
    train_x = numpy.delete(train_x, zero_ixs, axis=0)
    train_x_int = numpy.delete(train_x_int, zero_ixs, axis=0)
    train_x_str = numpy.delete(dataset.x_test, zero_ixs, axis=0)
    train_y = numpy.delete(train_y, zero_ixs, axis=0)
    train_y_str = numpy.delete(dataset.y_train, zero_ixs, axis=0)

    if do_train:
        model = build_model(max_sent_length, vocab, embeddings, num_classes)
        print(model.summary())

        # Calculate class weights
        all_labels: List[str] = [l for labels in train_y_str for l in labels]
        label_counts = Counter(all_labels)
        sum_labels_counts = sum(label_counts.values())
        class_weight = {numpy.where(mlb.classes_ == label)[0][0]: 1 - (cnt / sum_labels_counts) for label, cnt in
                        label_counts.items()}

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  patience=3, restore_best_weights=True)

        tensor_board = tensorflow.keras.callbacks.TensorBoard()

        model.fit(train_x, train_y, epochs=50, verbose=1,
                  validation_data=(dev_x, dev_y),
                  class_weight=class_weight, callbacks=[early_stopping, tensor_board])

        model.save('saved_models/%s' % model_name, overwrite=True)
    else:
        model = load_model('saved_models/%s' % model_name)

    # plot_model(model, to_file='/tmp/%s.png' % model_name)

    if do_test:
        y_pred_bin = model.predict(test_x)
        # TODO add classifier tuning
        y_pred = stringify_labels(y_pred_bin, mlb, thresh=classification_thresh)
        evaluate_multilabels(dataset.y_test, y_pred)

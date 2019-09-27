"""
Keras MLP classifier for provision classification using TF IDF weighted averaged MUSE embeddings
"""

import json
import pickle
import random; random.seed(42)
import numpy; numpy.random.seed(42)
import keras
from typing import List, Set, Dict, Tuple
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MultiLabelBinarizer
from utils import embed, SplitDataSet, split_corpus, stringify_labels, evaluate_multilabels, tune_clf_thresholds


def train(x_train, y_train, num_classes, batch_size, epochs, class_weight=None):
    print('Building model...')
    input_shape = x_train[0].shape[0]
    hidden_size_1 = input_shape * 2
    hidden_size_2 = int(input_shape / 2)
    model = Sequential()
    model.add(Dense(hidden_size_1, input_shape=(input_shape,), kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='relu'))
    model.add(Dropout(0.5, seed=42))
    model.add(Dense(hidden_size_2, kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='relu'))
    model.add(Dropout(0.5, seed=42))
    model.add(Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    print('Train model...')
    model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=2, validation_split=0., class_weight=class_weight)
    return model


if __name__ == '__main__':

    train_de = False
    test_de = True

    model_name = 'MLP_avg_tfidf_NDA.h5'

    epochs = 50
    batch_size = 32

    corpus_file = '../sec_corpus_2016-2019_clean_NDA_PTs.jsonl'

    print('Loading corpus', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')

    mlb = MultiLabelBinarizer().fit(dataset.y_train)
    num_classes = mlb.classes_.shape[0]
    train_y = mlb.transform(dataset.y_train)
    test_y = mlb.transform(dataset.y_test)

    embedding_file = '/home/don/resources/fastText_MUSE/wiki.multi.en.vec_data.npy'
    vocab_file = '/home/don/resources/fastText_MUSE/wiki.multi.en.vec_vocab.json'
    embeddings = numpy.load(embedding_file)
    vocab_de = json.load(open(vocab_file))
    print('Preprocessing')
    train_x = embed(dataset.x_train, embeddings, vocab_de, use_tfidf=True, avg_method='mean')
    test_x = embed(dataset.x_test, embeddings, vocab_de, use_tfidf=True, avg_method='mean')
    dev_x = embed(dataset.x_dev, embeddings, vocab_de, use_tfidf=True, avg_method='mean')

    # Calculate class weights
    all_labels: List[str] = [l for labels in dataset.y_train for l in labels]
    label_counts = Counter(all_labels)
    sum_labels_counts = sum(label_counts.values())
    class_weight = {numpy.where(mlb.classes_ == label)[0][0]: 1 - (cnt/sum_labels_counts) for label, cnt in label_counts.items()}

    if train_de:
        print('Training model')
        model = train(train_x, train_y, num_classes, batch_size, epochs, class_weight=class_weight)
        model.save('saved_models/%s' % model_name, overwrite=True)
    else:
        print('Loading model')
        model = keras.models.load_model('saved_models/%s' % model_name)

    y_pred_bin_dev = model.predict(dev_x)
    label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)
    y_pred_bin = model.predict(test_x)
    y_pred_thresh = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
    y_pred_nothresh = stringify_labels(y_pred_bin, mlb)
    print('MLP results without classifier threshold tuning')
    evaluate_multilabels(dataset.y_test, y_pred_nothresh, do_print=True)
    print('MLP results with classifier threshold tuning')
    evaluate_multilabels(dataset.y_test, y_pred_thresh, do_print=True)


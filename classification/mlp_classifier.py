"""
Keras MLP classifier for provision classification using TF IDF weighted averaged MUSE embeddings
"""

import json
import random; random.seed(42)
import numpy; numpy.random.seed(42)
import tensorflow.keras as keras
from typing import List
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import MultiLabelBinarizer
from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds


def build_model(x_train, num_classes):
    print('Building model...')
    input_shape = x_train[0].shape[0]
    hidden_size = input_shape * 2
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_shape,), activation='relu'))
    # model.add(Dropout(0.5, seed=42))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':

    train_de = False
    test_de = False
    use_tfidf = False
    test_nda = True

    # model_name = 'MLP_avg_NDA_prop.h5'
    # corpus_file = '../nda_proprietary_data2_sampled.jsonl'

    # model_name = 'MLP_avg_proto.h5'
    # corpus_file = 'data/sec_corpus_2016-2019_clean_proto.jsonl'

    # model_name = 'MLP_avg_leaves.h5'
    # corpus_file = 'data/sec_corpus_2016-2019_clean_projected_real_roots.jsonl'

    # model_name = 'MLP_avg_freq100.h5'
    # corpus_file = 'data/sec_corpus_2016-2019_clean_freq100.jsonl'

    # model_name = 'MLP_avg_leaves_subsampled.h5'
    # corpus_file = 'data/sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'

    model_name = 'MLP_avg_NDA.h5'
    corpus_file = 'data/sec_corpus_2016-2019_clean_NDA_PTs2.jsonl'

    epochs = 50
    batch_size = 32

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

    embedding_file = 'data/wiki.multi.en.vec_data.npy'
    vocab_file = 'data/wiki.multi.en.vec_vocab.json'
    embeddings = numpy.load(embedding_file)
    vocab_en = json.load(open(vocab_file))
    print('Preprocessing')
    train_x = embed(dataset.x_train, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')
    test_x = embed(dataset.x_test, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')
    dev_x = embed(dataset.x_dev, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')

    """
    # Calculate class weights
    all_labels: List[str] = [l for labels in dataset.y_train for l in labels]
    label_counts = Counter(all_labels)
    sum_labels_counts = sum(label_counts.values())
    # TODO use sklearn's method to calculate class weights
    class_weight = {numpy.where(mlb.classes_ == label)[0][0]:
                        1 - (cnt/sum_labels_counts)
                    for label, cnt in label_counts.items()}
    """

    if train_de:
        model = build_model(train_x, num_classes)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       mode='min', verbose=1,
                                       patience=3, restore_best_weights=True)
        print('Train model...')
        try:
            model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
                      verbose=1, validation_data=(dev_x, dev_y),
                      # class_weight=class_weight,
                      callbacks=[early_stopping])
        except KeyboardInterrupt:
            pass
        model.save('saved_models/%s' % model_name, overwrite=True)

    else:
        print('Loading model')
        model = keras.models.load_model('saved_models/%s' % model_name)

    if test_de:
        y_pred_bin_dev = model.predict(dev_x)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb, objective='f1')
        y_pred_bin = model.predict(test_x)
        y_pred_thresh = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        print('MLP results with classifier threshold tuning')
        res = evaluate_multilabels(dataset.y_test, y_pred_thresh, do_print=True)

    if test_nda:
        nda_file = 'data/nda_proprietary_data2_sampled.jsonl'
        print('Loading corpus from', nda_file)
        dataset_nda: SplitDataSet = split_corpus(nda_file)
        train_y = mlb.transform(dataset_nda.y_train)
        test_y = mlb.transform(dataset_nda.y_test)
        dev_y = mlb.transform(dataset_nda.y_dev)

        nda_x_train = embed(dataset_nda.x_train, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')
        nda_x_test = embed(dataset_nda.x_test, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')
        nda_x_dev = embed(dataset_nda.x_dev, embeddings, vocab_en, use_tfidf=use_tfidf, avg_method='mean')

        y_preds_nda_probs_dev = model.predict(nda_x_dev)
        label_threshs = tune_clf_thresholds(y_preds_nda_probs_dev, dataset_nda.y_dev, mlb)
        y_preds_nda_probs_test = model.predict(nda_x_test, verbose=1)
        y_preds_nda = stringify_labels(y_preds_nda_probs_test, mlb, label_threshs=label_threshs)

        print('MLP results NDA with classifier threshold tuning')
        evaluate_multilabels(dataset_nda.y_test, y_preds_nda, do_print=True)
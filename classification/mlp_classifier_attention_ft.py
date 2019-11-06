"""
Sentence classification with self-attention in keras
"""

import json
import numpy; numpy.random.seed(42)
import re
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import fasttext
from attn_layer import AttentionLayer
from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds


def build_model(max_sent_length, embedding_dim, num_labels):

    input_layer = Input(shape=(max_sent_length, embedding_dim,))

    l_att = AttentionLayer()(input_layer)

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

    # corpus_file = 'data/sec_corpus_2016-2019_clean_freq100_subsampled.jsonl'
    # model_name = 'MLP_attn_freq100_subsampled.h5'

    # corpus_file = 'data/sec_corpus_2016-2019_clean_proto.jsonl'
    # model_name = 'MLP_attn_proto.h5'

    # corpus_file = 'data/sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'
    # model_name = 'MLP_attn_leaves_subsampled.h5'

    corpus_file = 'data/sec_corpus_2016-2019_clean_NDA_PTs2.jsonl'
    model_name = 'MLP_attn_nda_ft.h5'

    # corpus_file = 'data/nda_proprietary_data2_sampled.jsonl'
    # model_name = 'MLP_attn_nda_prop_ft.h5'

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

    print('Loading embeddings')
    embedding_file = 'data/cc.en.300.bin'
    embeddings = fasttext.load_model(embedding_file)

    print('Vectorizing training set')
    train_x_int = [[embeddings.get_word_vector(w) for w in re.findall('\w+', x_.lower())]
                   for x_ in dataset.x_train]
    print('Vectorizing test set')
    test_x_int = [[embeddings.get_word_vector(w) for w in re.findall('\w+', x_.lower())]
                  for x_ in dataset.x_test]
    print('Vectorizing dev set')
    dev_x_int = [[embeddings.get_word_vector(w) for w in re.findall('\w+', x_.lower())]
                 for x_ in dataset.x_dev]

    max_sent_length = max([len(x_) for x_ in train_x_int])
    train_x = pad_sequences(train_x_int, max_sent_length)
    test_x = pad_sequences(test_x_int, max_sent_length)
    dev_x = pad_sequences(dev_x_int, max_sent_length)

    if do_train:
        embedding_dim = 300
        model = build_model(max_sent_length, embedding_dim, num_classes)
        print(model.summary())

        # TODO calculate weights in sklearn fashion
        """
        # Calculate class weights
        all_labels: List[str] = [l for labels in train_y_str for l in labels]
        label_counts = Counter(all_labels)
        sum_labels_counts = sum(label_counts.values())
        class_weight = {numpy.where(mlb.classes_ == label)[0][0]: 1 - (cnt / sum_labels_counts) for label, cnt in
                        label_counts.items()}
        """

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       restore_best_weights=True)

        try:
            model.fit(train_x, train_y,
                      batch_size=32,
                      epochs=50,
                      verbose=1,
                      validation_data=(dev_x, dev_y),
                      # class_weight=class_weight,
                      callbacks=[early_stopping])
        except KeyboardInterrupt:
            pass

        model.save('saved_models/%s' % model_name, overwrite=True)

    else:
        model = load_model('saved_models/%s' % model_name,
                           custom_objects={'AttentionLayer': AttentionLayer})

    # plot_model(model, to_file='/tmp/%s.png' % model_name)

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

        train_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                       for x_ in dataset.x_train]
        test_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                      for x_ in dataset.x_test]
        dev_x_int = [[vocab[w] for w in re.findall('\w+', x_.lower()) if w in vocab]
                     for x_ in dataset.x_dev]

        train_x = pad_sequences(train_x_int, max_sent_length)
        test_x = pad_sequences(test_x_int, max_sent_length)
        dev_x = pad_sequences(dev_x_int, max_sent_length)

        # Remove zero-valued training examples and labels; they break fit()!
        zero_ixs = [i for i, x_ in enumerate(train_x_int) if not x_]
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

        print('predicting NDA')
        y_pred_bin_dev = model.predict(dev_x, verbose=1)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)
        y_pred_bin = model.predict(test_x, verbose=1)
        y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

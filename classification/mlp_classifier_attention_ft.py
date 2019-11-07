"""
Sentence classification with self-attention in keras
"""

import json
import numpy; numpy.random.seed(42)
import re
import math
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


def data_generator(x, y, max_sent_length, embeddings, batch_size=32):
    x_out, y_out = [], []
    while True:
        for i, (x_, y_) in enumerate(zip(x, y), start=1):
            x_int = [embeddings.get_word_vector(w) for w in re.findall('\w+', x_.lower())]
            x_out.append(x_int)
            y_out.append(y_)
            if i > 1 and i % batch_size == 0:
                x_out = pad_sequences(x_out, max_sent_length)
                yield ({'input_1': x_out}, {'dense_2': numpy.array(y_out)})
                x_out, y_out = [], []


if __name__ == '__main__':

    do_train = True
    do_test = True
    do_test_nda = False
    classification_thresh = 0.5
    batch_size = 32

    # corpus_file = 'data/sec_corpus_2016-2019_clean_freq100_subsampled.jsonl'
    # model_name = 'MLP_attn_freq100_subsampled_ft.h5'

    # corpus_file = 'data/sec_corpus_2016-2019_clean_proto.jsonl'
    # model_name = 'MLP_attn_proto_ft.h5'

    # corpus_file = 'data/sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'
    # model_name = 'MLP_attn_leaves_subsampled_ft.h5'

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

    max_sent_length = max(
        [
            len([w for w in re.findall('\w+', x_.lower())])
            for x_ in dataset.x_train
         ]
    )

    if do_train:
        embedding_dim = 300  # SIze of word embeddings
        model = build_model(max_sent_length, embedding_dim, num_classes)
        print(model.summary())

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       restore_best_weights=True)

        try:
            # step_size = math.ceil(len(dataset.y_train) / batch_size)
            step_size = len(dataset.x_train)  # No. of samples per epoch
            model.fit_generator(data_generator(dataset.x_train, train_y, max_sent_length, embeddings,
                                               batch_size=batch_size),
                                steps_per_epoch= step_size,
                                epochs=50,
                                verbose=1,
                                validation_data=data_generator(dataset.x_dev, dev_y, max_sent_length, embeddings,
                                                               batch_size=batch_size),
                                validation_steps=len(dataset.x_dev),
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
        y_pred_bin_dev = model.predict_generator(data_generator(dataset.x_dev, dev_y, max_sent_length, embeddings,
                                                     batch_size=batch_size), verbose=1)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)

        y_pred_bin = model.predict(data_generator(dataset.x_test, test_y, max_sent_length, embeddings,
                                                     batch_size=batch_size), verbose=1)

        y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

    if do_test_nda:
        nda_file = 'data/nda_proprietary_data2_sampled.jsonl'
        print('Loading corpus from', nda_file)
        dataset: SplitDataSet = split_corpus(nda_file)

        train_y = mlb.transform(dataset.y_train)
        test_y = mlb.transform(dataset.y_test)
        dev_y = mlb.transform(dataset.y_dev)

        print('predicting NDA')
        y_pred_bin_dev = model.predict(dev_x, verbose=1)
        label_threshs = tune_clf_thresholds(y_pred_bin_dev, dataset.y_dev, mlb)
        y_pred_bin = model.predict(test_x, verbose=1)
        y_pred = stringify_labels(y_pred_bin, mlb, label_threshs=label_threshs)
        evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

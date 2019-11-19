import numpy
from fastai.text import *
from utils import SplitDataSet, split_corpus, evaluate_multilabels


def write_csv(x, y, outfile, lowercasing=False):
    with open(outfile, 'w') as f:
        for text, labels in zip(x, y):
            label_str = ','.join(labels)
            if lowercasing:
                text = text.lower()
            f.write(label_str + '\t' + text + '\n')


def preds_to_labels(learn, preds):
    labels = []
    ix2label = {i: l for l, i in learn.data.c2i.items()}
    for sample in preds:
        fired = numpy.where(sample >= 0.5)[0]
        sample_labels = [ix2label[i] for i in fired]
        labels.append(sample_labels)
    return labels


if __name__ == '__main__':

    # TODO
    # threshold tuning
    # class_weights

    corpus_file = '../nda_proprietary_data_sampled.jsonl'
    classifier_file = 'fastai_NDA_prop.mdl'

    lowercasing = False
    batch_size = 16
    tmp_path = '/tmp'
    re_process = False
    retrain_lm = False
    retrain_clf = False

    print('Loading corpus from', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')

    label_set = set(l for labels in dataset.y_train for l in labels)
    print('Label set size:', len(label_set))

    train_file = 'train_' + os.path.basename(corpus_file)
    test_file = 'test_' + os.path.basename(corpus_file)
    dev_file = 'dev_' + os.path.basename(corpus_file)

    write_csv(dataset.x_train, dataset.y_train, os.path.join(tmp_path, train_file), lowercasing=lowercasing)
    write_csv(dataset.x_test, dataset.y_test, os.path.join(tmp_path, test_file), lowercasing=lowercasing)
    write_csv(dataset.x_dev, dataset.y_dev, os.path.join(tmp_path, dev_file), lowercasing=lowercasing)

    if re_process:
        data_lm = TextLMDataBunch.from_csv(tmp_path, train_file,
                                           label_delim=',', test=test_file, delimiter='\t')
        data_clas = TextClasDataBunch.from_csv(tmp_path, train_file,
                                               test=test_file, delimiter='\t', label_delim=',',
                                               vocab=data_lm.train_ds.vocab, bs=batch_size)
        data_lm.save('data_lm_export.pkl')
        data_clas.save('data_clas_export.pkl')
    else:
        data_lm = load_data('/tmp', 'data_lm_export.pkl')
        data_clas = load_data('/tmp', 'data_clas_export.pkl', bs=batch_size)

    # Language model fine-tuning on the training data
    if retrain_lm:
        learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.25)
        learn_lm.unfreeze()
        learn_lm.lr_find()
        learn_lm.fit_one_cycle(1, 1e-2)
        learn_lm.save_encoder('ft_enc')
    # learn_lm.predict("Confidential information should", n_words=10)

    # Classifier
    if retrain_clf:
        learn_clf = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.25)
        learn_clf.load_encoder('ft_enc')
        learn_clf.unfreeze()
        # learn_clf.lr_find()
        learn_clf.fit_one_cycle(1, 1e-2)
        breakpoint()
        learn_clf.export(classifier_file)
    else:
        learn_clf = load_learner(classifier_file)

    preds, y = learn_clf.get_preds(data_clas.test_ds)
    pred_labels = preds_to_labels(learn_clf, preds)
    y_labels = preds_to_labels(learn_clf, y)
    evaluate_multilabels(y_labels, pred_labels, do_print=True)

    breakpoint()

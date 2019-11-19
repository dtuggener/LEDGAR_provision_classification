import os
import fasttext
from utils import embed, SplitDataSet, split_corpus, stringify_labels, \
    evaluate_multilabels, tune_clf_thresholds


def write_fasttext_file(x, y, outfile, lowercasing=False):
    with open(outfile, 'w') as f:
        for text, labels in zip(x, y):
            label_str = ' '.join(['__label__' + label for label in labels])
            if lowercasing:
                text = text.lower()
            f.write(label_str + '\t"' + text + '"\n')


def predict(model, texts, lowercasing=False):
    y_pred, y_probs = [], []
    for text in texts:
        if lowercasing:
            text = text.lower()
        labels, probs = model.predict(text)
        labels = [l.replace('__label__', '') for l in labels]
        y_pred.append(labels)
        y_probs.append(probs)
    return y_pred, y_probs


if __name__ == '__main__':

    do_train = True
    lowercasing = True

    corpus_file = 'data/sec_corpus_2016-2019_clean_NDA_PTs2.jsonl'
    classifier_file = 'saved_models/fasttext_sec_clf_nda.mdl'
    # corpus_file = 'data/sec_corpus_2016-2019_clean_freq100.jsonl'
    # classifier_file = 'saved_models/fasttext_sec_clf_freq100.mdl'
    #corpus_file = '../sec_corpus_2016-2019_clean_projected_real_roots_subsampled.jsonl'
    #classifier_file = 'saved_models/fasttext_sec_clf_leaves_subsampled.mdl'


    print('Loading corpus from', corpus_file)
    dataset: SplitDataSet = split_corpus(corpus_file)
    print(len(dataset.y_train), 'training samples')
    print(len(dataset.y_test), 'test samples')
    print(len(dataset.y_dev), 'dev samples')
    label_set = set(l for labels in dataset.y_train for l in labels)
    print('Label set size:', len(label_set))

    train_file = '/tmp/train' + os.path.basename(corpus_file)
    test_file = '/tmp/test' + os.path.basename(corpus_file)
    dev_file = '/tmp/dev' + os.path.basename(corpus_file)

    print('Writing tmp files')
    write_fasttext_file(dataset.x_train, dataset.y_train, train_file, lowercasing=lowercasing)
    write_fasttext_file(dataset.x_test, dataset.y_test, test_file, lowercasing=lowercasing)
    write_fasttext_file(dataset.x_dev, dataset.y_dev, dev_file, lowercasing=lowercasing)

    if do_train:
        print('Training')
        model = fasttext.train_supervised(input=train_file, epoch=50, bucket=200000, loss='ova')
        model.save_model(classifier_file)
    else:
        model = fasttext.load_model(classifier_file)

    res = model.test(test_file)
    print(res)
    y_pred, _ = predict(model, dataset.x_test, lowercasing=lowercasing)
    evaluate_multilabels(dataset.y_test, y_pred, do_print=True)

    # TODO clf thresh tuning
    # y_pred_dev, y_prob_dev = predict(model, dataset.x_dev, lowercasing=lowercasing)
    # breakpoint()







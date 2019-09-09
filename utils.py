import json
from typing import List, Union
from sklearn.model_selection import train_test_split


# TODO use 3.7 dataclass for this
class DataSet():
    def __init__(self, x: List[str], y: List[List[str]], use_dev: bool = True):
        x: List[str] = []
        y: List[List[str]] = []
        doc_ids: List[str] = []

        for line in open(corpus_file):
            labeled_provision = json.loads(line)
            x.append(labeled_provision['provision'])
            y.append(labeled_provision['label'])
            doc_ids.append(labeled_provision['source'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        if use_dev:
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        else:
            x_dev, y_dev = None, None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_dev = x_dev
        self.y_dev = y_dev



def split_data(corpus_file: str, use_dev: bool=True):




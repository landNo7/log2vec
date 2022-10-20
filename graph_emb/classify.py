from __future__ import print_function


import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = clf
        # self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.binarizer = LabelBinarizer()

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        # top_k_list = [len(l) for l in Y]
        # Y_ = self.predict(X, top_k_list)
        # Y = self.binarizer.transform(Y)
        Y_ = self.predict(X)
        pred = self.predict_proba(X)[:, 1]
        results = {}
        results['y_test'] = Y
        Y = self.binarizer.transform(Y)

        # averages = ["micro", "macro", "samples", "weighted"]
        # for average in averages:
        #     results[average] = f1_score(Y, Y_, average=average)

        results['acc'] = accuracy_score(Y,Y_)
        results['pred'] = pred
        results['x_test'] = X
        results['y_test_'] = Y
        results['y_pred'] = Y_
        print('-------------------')
        print("acc:", results['acc'])
        print('-------------------')
        return results

    # def predict(self, X, top_k_list):
    def predict(self, X, top_k_list=None):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        # Y = self.clf.predict(X_, top_k_list=top_k_list)
        Y = self.clf.predict(X_)
        return Y

    def predict_proba(self, X):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict_proba(X_)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

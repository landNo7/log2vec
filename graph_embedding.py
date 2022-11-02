import pandas as pd
import numpy as np

from graph_emb.classify import read_node_label, Classifier
from graph_emb import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_curve, auc
from label_processing import *
from tqdm import tqdm
import time


def evaluate_embeddings(clf, dirpath, filename):
    X, Y = get_label(dirpath, filename, "anomaly_list.txt")
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    results = clf.split_train_evaluate(X, Y, tr_frac)
    print(clf.predict_proba(["10.146.236.10"]))
    return results


def plot_embeddings(embeddings, dirpath, filename):
    X, Y = get_label(dirpath, filename, "anomaly_list.txt")

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in tqdm(range(len(X))):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in tqdm(color_idx.items()):
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

def plot_auc(y_test, pred):
    fpr, tpr, threshold = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_tpfp(y_pred, y_test):
    
    y_test = np.array(y_test).flatten()
    res = (y_pred ^ y_test)
    r = np.bincount(res)
    tp_list = ((y_pred) & (y_test))
    fp_list = ((y_pred) & ~(y_test))
    tp_list = tp_list.tolist()
    fp_list = fp_list.tolist()
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = r[0] - tp
    fn = r[1] - fp
    return tp, fp, tn, fn

def predict(X, islist=1):
    pred = clf.predict_proba(X)
    pred = np.array(pred).flatten()
    pred = pred.reshape(-1, 2)
    X = np.array(X)
    X = X.reshape(-1, 1)
    return np.concatenate([X, pred], axis=1)

csvname = "attack"
gpickle_name = "./data/{csvname}_graph.gpickle".format(csvname=csvname)
print("...read {gpickle_name} graph start...".format(gpickle_name=gpickle_name))
rg_st_time = time.time()
G = nx.read_gpickle(gpickle_name)
print("...read {gpickle_name} graph completed, time cost:".format(gpickle_name=gpickle_name), time.time() - rg_st_time)

print("...read model start...")
train_st_time = time.time()
# 序列长度，xxx，并行worker数量
model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
print("...read model completed, time cost:", time.time() - train_st_time)

print("...model train start...")
train_st_time = time.time()
model.train(window_size=5, iter=3) 
print("...model train completed, time cost:", time.time() - train_st_time)

print("...get model embeddings...")
get_e_st_time = time.time()
embeddings = model.get_embeddings()
print("...got model embeddings, time cost:", time.time() - get_e_st_time)
train_X = []
train_X_id = []

for k, v in tqdm(embeddings.items()):
    train_X.append(v)
    train_X_id.append(v)

train_X = np.array(train_X)

print("...DBSCAN fit start...")
fit_st_time = time.time()
clustering = DBSCAN().fit(train_X)
print("...DBSCAN fit completed, time cost:", time.time() - fit_st_time)

# print("...embeddings print...")
# print(embeddings)
# print("...embeddings end...")

# print("...train_X print...")
# print(train_X)
# print("...train_X end...")

print("...evaluate_embeddings start...")
ee_time = time.time()
clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
results = evaluate_embeddings(clf, "./data", csvname + ".csv")
print("...evaluate_embeddings completed time cost:", time.time() - ee_time)

print("...code run cost:", time.time() - rg_st_time)

pd.DataFrame(predict(get_all_attackip("./data", csvname + ".csv"))).to_csv('./data/results.csv')
# print(clf.predict_proba(["10.146.236.10"]))
# tp, fp, tn, fn = get_tpfp(results['y_pred'], results['y_test_'])
# print("tp=%d, fp=%d, tn=%d, fn=%d" % (tp, fp, tn, fn))
# plot_auc(results['y_test_'], results['pred'])
# plot_embeddings(embeddings, "./data", csvname + ".csv")



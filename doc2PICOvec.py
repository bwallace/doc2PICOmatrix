
''' Needs to be run under Python 3.x! '''

import sys
import pdb 

import numpy as np
import scipy as sp

import pandas as pd  

import sklearn 
from sklearn import manifold # for TSNE
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim 
from gensim.models import Doc2Vec

import matplotlib.pyplot as plt 
import seaborn as sns 

'''
infer_vector is what you want.
'''


def load_trained_model(path="Doc2Vec/400_pvdbow_doc2vec.d2v"):
    ''' @TODO swap in MEDLINE trained variant '''
    m = Doc2Vec.load(path)
    return m

# demo/proof of concept
def TNSE(d2v=None):
    pmids, X, y = PICO_embed(d2v=d2v)
    tsne = manifold.TSNE()

    X_pr = tsne.fit_transform(X)

    X_pr_pos = X_pr[y>0]
    X_pr_neg = X_pr[y<=0]

    plt.scatter(X_pr_neg[:,0], X_pr_neg[:,1], c="red", alpha=.3, edgecolors='none')
    plt.scatter(X_pr_pos[:,0], X_pr_pos[:,1], c="blue", edgecolors='none')

    plt.show()


def PICO_embed(path="tagged_data/cohen/ACEInhibitors_processed_PICO.tsv", d2v=None):
    if d2v is None:
        # optionally allow to be passed in
        d2v = load_trained_model() # traine doc2vec model 

    PICO_data = pd.read_csv(path, sep="\t")

    vecs, y, pmids = [], [], []

    for p, i, o, lbl, pmid in PICO_data[["P", "I", "O", "y", "pmid"]].values:
        pmids.append(pmid)

        pv = d2v.infer_vector(p)         
        pv = pv - pv.min()
        iv = d2v.infer_vector(i) 
        iv = iv - iv.min() 
        ov = d2v.infer_vector(o)
        ov = ov - ov.min()
        #pdb.set_trace()

        '''
        note to self; i actually think a PICO matrix 
        would be better for CNNs; this would mean we
        could have interactions/convolutions over the
        P/I and I/O and P/I/O (just using different heights)
        '''
        PICO_vec = np.hstack((pv, iv, ov))
        vecs.append(PICO_vec)

        y.append(lbl)

    return pmids, np.array(vecs), np.array(y)


'''
@TODO all classificaiton stuff should be factored out of 
this module!!!
'''
def load_texts(path="input_data/cohen/ACEInhibitors_processed.csv"):
    citation_data = pd.read_csv(path, header=None, 
                    names =["pmid", "title", "authors", "journal", "abstract", "keywords", "label"])
    return citation_data


def get_tfidf_X_y(path="input_data/cohen/ACEInhibitors_processed.csv"):
    citation_data = load_texts(path=path) 
    pmids, texts = [], []
    for pmid, title, abstract, mesh in citation_data[["pmid", "title", "abstract", "keywords"]].values:
        pmids.append(pmid)
        mesh_words = " ".join(["MH_%s" % mh_term.strip().replace("-", "_") for mh_term in mesh.split(",")])
        #mesh_words = " "
        title_words = " ".join(["TI_%s" % w.strip() for w in title.split(" ")])
        texts.append(" ".join((title_words, abstract, mesh_words)))

    v = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=50000)
    X_text = v.fit_transform(texts)
    return dict(zip(pmids, X_text)), v

def get_naive_embedding(path="input_data/cohen/ACEInhibitors_processed.csv"):
    d2v = load_trained_model() 

    citation_data = load_texts(path=path) 
    pmids, texts, vecs = [], [], []
    for pmid, title, abstract, mesh in citation_data[["pmid", "title", "abstract", "keywords"]].values:
        pmids.append(pmid)
        cur_text = " ".join((title, abstract))
        texts.append(cur_text)
        vecs.append(d2v.infer_vector(cur_text))
        #pdb.set_trace()

    #TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=50000)
    #X_text = v.fit_transform(texts)
    #return dict(zip(pmids, X_text)), v
    return dict(zip(pmids, vecs))

def run_exp_for_X_y(X, y, model="SGD"):
    if model == "SGD":    
        alpha_range = 10.0**-np.arange(1,7)
        m = SGDClassifier()
        parameters = {"alpha":alpha_range}
    else:
        m = sklearn.svm.SVC()
        C_range = np.logspace(-2, 10, 10)
        gamma_range = np.logspace(-9, 3, 10)
        parameters = dict(gamma=gamma_range, C=C_range)

    clf = sklearn.grid_search.GridSearchCV(m, parameters, scoring='roc_auc')
    #skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=50)
    res = cross_validation.cross_val_score(clf, X, y, scoring="roc_auc", cv=5)
    return res 

def classification_exp():
    pmids, X, y = PICO_embed()
    res = run_exp_for_X_y(X,y)
    run_exp_for_X_y(X, y)

    # now with text 
    X_tfidf_d, v = get_tfidf_X_y()
    X2 = []
    for pmid in pmids:
        X2.append(X_tfidf_d[pmid])

    X2 = sp.sparse.vstack(X2)
    res2 = run_exp_for_X_y(X2, y)

    # now with both
    X3 = []
    for i in range(X.shape[0]):
        X3.append(sp.sparse.hstack((X[i,:], X2[i,:])))
    X3 = sp.sparse.vstack(X3)
    res3 = run_exp_for_X_y(X3, y)

    # one more!
    X_embedded_d = get_naive_embedding()
    X4 = []
    for pmid in pmids:
        X4.append(X_embedded_d[pmid])
    #pdb.set_trace()
    X4 = np.vstack(X4) #sp.sparse.vstack(X4)
    #pdb.set_trace()
    res4 = run_exp_for_X_y(X4, y)

    # one more!
   
    X5 = []
    for i in range(X.shape[0]):
        X5.append(sp.sparse.hstack((X[i,:], X4[i,:])))

    #pdb.set_trace()
    X5 = sp.sparse.vstack(X5)
    res5 = run_exp_for_X_y(X5, y)

    return res, res2, res3, res4, res5


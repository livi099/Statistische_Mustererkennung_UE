# Statistische Muserekennung WS 2023
# Laurin Koppenwallner, 11726954
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import random

# Aufgabe 1


def kNN(X_tr, y_tr, X_te, k):
    result=[]


    return

def dist(p1, p2):
    d = np.sqrt((p2(0)-p1(0))**2+(p2(1)-p1(1))**2)
    return(d)

def build_testtrain(mk, lb, n, tr_size):
    idx = np.random.permutation(len(mk))
    train_size = int(len(mk) * tr_size)
    idx_train = idx[:train_size]
    idx_test = idx[train_size:]
    X_tr = mk[idx_train] # Merkmal training
    y_tr = lb[idx_test] # Label training
    X_te = mk[idx_test]
    y_te = lb[idx_test] # für Korrelation
    return(X_tr, y_tr, X_te, y_te)



# load input files
mk = np.loadtxt('data/perceptrondata.txt')
lb = np.loadtxt('data/perceptrontarget2.txt')

# Anzahl Wiederholungen
n = 5

size_train = 0.5 # beide Mengen gleich groß

np.random.seed(0)

i = 0
while i < n:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, n, size_train)



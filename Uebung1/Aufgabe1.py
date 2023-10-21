# Statistische Muserekennung WS 2023
# Laurin Koppenwallner, 11726954
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Aufgabe 1

def kNN(X_tr, y_tr, X_te, k):

    test_points=[]
    for x_test in X_te:
        distances = [dist(x_test, x_train) for x_train in X_tr] # Berechne von dem aktuellen Testpunkt die Distanzen zu allen Trainingspunkten
        df = pd.DataFrame({'distances': distances, 'labels': y_te}) # Füge die Distanzen und die Labels der Trainigspunkte zu einem Dataframe zusammen
        df.sort_values(by='distances', ascending=True) # Sortiere das Dataframe nach den Distanzen aufsteigend
        t = df[:k]
        if (t['labels'] == 0).sum() > (t['labels'] == 1).sum():
            test_points.append(0)

        else:
            test_points.append(1)

    df2 = pd.DataFrame({'x': X_te[:,0], 'y': X_te[:,1], 'labels': test_points})


    return(df2)

def dist(p1, p2):
    d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
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

def validate_test_points(test_p, y_te):
    acc = (sum(test_p['labels'] == y_te) / len(y_te) * 100)
    return(acc)



# Lade files
mk = np.loadtxt('data/perceptrondata.txt')
lb = np.loadtxt('data/perceptrontarget2.txt')

# Anzahl Datensets
n = 5

size_train = 0.5 # beide Mengen gleich groß

np.random.seed(1234)

k = list(range(1, 21, 2))

results = pd.DataFrame({'k':k})
arr = []
i = 0
while i < n:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, n, size_train)
    arr = []
    for j in k:
        test_p = kNN (X_tr, y_tr, X_te, j)
        corr = validate_test_points(test_p, y_te)
        arr.append(corr)
    results['dataset%s'%i]=arr
    i = i + 1
print(results)

# Plot
plt.plot(results['k'],results['dataset0'])
plt.plot(results['k'],results['dataset1'])
plt.plot(results['k'],results['dataset2'])
plt.plot(results['k'],results['dataset3'])
plt.plot(results['k'],results['dataset4'])
plt.xticks(k)
plt.xlabel('k')
plt.ylabel('Genauigkeit [%]')
plt.grid()
plt.show()

data = np.array([results['k'], results['dataset0']])
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=["k-Values", "Performance"])
plt.xlabel("k")
plt.title("Performance Heatmap for Different k-Values")
plt.show()



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

        df = pd.DataFrame({'distances': distances, 'labels': y_tr}) # Füge die Distanzen und die Labels der Trainigspunkte zu einem Dataframe zusammen
        df = df.sort_values(by='distances', ascending=True) # Sortiere das Dataframe nach den Distanzen aufsteigend
        t = df[:k]


        if (t['labels'] == 0).sum() > (t['labels'] == 1).sum():
            test_points.append(int(0))
        else:
            test_points.append(int(1))

    df['test_points'] = test_points

    return(df)

def dist(p1, p2):
    d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return(d)

def build_testtrain(mk, lb, tr_size, seed_num):

    random.seed(i_seed)
    count = round(len(mk) * tr_size)
    idx = random.sample(range(len(mk)),len(mk))
    train_size = int(len(mk) * tr_size)
    idx_train = idx[:train_size]
    idx_test = idx[train_size:]

    # idx_train = random.sample(range(len(mk)), 100)
    # idx_test = [x for x in range(len(mk)) if x not in idx_train]

    X_tr = mk[idx_train] # Merkmal training
    y_tr = lb[idx_train] # Label training
    X_te = mk[idx_test]
    y_te = lb[idx_test] # für Korrelation

    return(X_tr, y_tr, X_te, y_te)


def validate_test_points(test_p, y_te):


    acc = (sum(y_te == test_p['test_points']) / len(y_te) * 100)

    return(acc)




# Lade files
mk = np.loadtxt('data/perceptrondata.txt')
lb = np.loadtxt('data/perceptrontarget2.txt')

lb=lb.astype(int)



n = 5 # Anzahl Datensets
size_train = 0.5 # Größe der Trainingsmenge
k = list(range(1, 21, 2))

random.seed(1234)
random_seeds = random.sample(range(10000000), n)

results = pd.DataFrame({'k':k})
arr = []
i = 0
#while i < n:
for i_seed in random_seeds:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, size_train, i_seed)
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



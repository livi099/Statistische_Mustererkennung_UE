# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# Aufgabe 1

def kNN(X_tr, y_tr, X_te, k):
    """
    Klassifiziert aus den k-nächsten Nachbarn einen Testwert

    Parameters
    ----------
    X_tr : 2D array

    y_tr : 1D array

    X_te : 2D array

    k : int

    Return
    ----------
    df_results : dataframe
    """

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

    df_results = pd.DataFrame({'test_points': test_points})
    # df['test_points'] = test_points

    return(df_results)

def dist(p1, p2):
    """
    Berechnet die Distanz zwischen zwei Punkten

    Parameters
    ----------
    p1 : 2D array
        Points

    p2 : 2D array
        Points

    Return
    ----------
    d : float

    """
    d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

    return(d)

def build_testtrain(mk, lb, tr_size, seed_num):
    """
    Bildet aus gegebenen Daten zufällige Tainings- und Testmengen

    Parameters
    ----------
    mk : 2D array
        Merkmale

    lb : 1D array
        Labels

    tr_size : float

    seed_num : int

    Return
    ----------
    X_tr : 2D array
    y_tr : 1D array
    X_te : 2D array
    y_te : 1D array
    """
    random.seed(seed_num)
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
    """
    Vergleicht die Übereinstimmung der "originalen" Labels mit den neuen Labels und berechnet die Genauigkeit

    Parameters
    ----------
    test_p : 1D array

    y_te : 1D array

    Return
    ----------
    acc : 1D array
    """

    acc = (sum(y_te == test_p['test_points']) / len(y_te) * 100)

    return(acc)

def plot_line(data, name):
    """
    Erstellt einen Lineplot

    Parameters
    ----------
    data : dataframe

    name : 1D array
    """

    plt.plot(data['k'], data['dataset0'], label="Dataset1")
    plt.plot(data['k'], data['dataset1'], label="Dataset2")
    plt.plot(data['k'], data['dataset2'], label="Dataset3")
    plt.plot(data['k'], data['dataset3'], label="Dataset4")
    plt.plot(data['k'], data['dataset4'], label="Dataset5")

    if len(data.columns) == 11:
        plt.plot(data['k'], data['dataset5'], label="Dataset6")
        plt.plot(data['k'], data['dataset6'], label="Dataset7")
        plt.plot(data['k'], data['dataset7'], label="Dataset8")
        plt.plot(data['k'], data['dataset8'], label="Dataset9")
        plt.plot(data['k'], data['dataset9'], label="Dataset10")
    plt.xticks(k)
    plt.xlabel('k')
    plt.ylabel('Genauigkeit [%]')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=5, fancybox=True, prop={'size': 9})
    plt.grid()
    # plt.savefig('plots/Aufgabe1/line/line_dataset' + name + '.eps', format='eps')
    plt.show()


def plot_mean(data, name):
    """
    Erstellt einen Lineplot

    Parameters
    ----------
    data : dataframe

    name : string
    """
    mean = []
    std = []
    for i in range(len(data)):
        mean_row = data.iloc[i, 1:6].mean()
        std_row = data.iloc[i, 1:6].std()
        mean.append(mean_row)
        std.append(std_row)

    data['Mittelwert_pro_Reihe'] = mean
    data['Standardabweichung_pro_Reihe'] = std

    max_wert = data['Mittelwert_pro_Reihe'].max()
    id_max_wert = data['Mittelwert_pro_Reihe'].idxmax()
    k_max_wert = data.loc[id_max_wert, 'k']

    plt.errorbar(data['k'], data['Mittelwert_pro_Reihe'], yerr=data['Standardabweichung_pro_Reihe'], fmt='-o', ms=5, label ='Mittelwert mit Standardabweichung', zorder=1)
    plt.scatter(k_max_wert, max_wert, marker="*", color='red', s = 80, label="Optimaler Wert", zorder = 2)
    plt.legend(loc='lower right')
    plt.xticks(data['k'])
    plt.xlabel('k')
    plt.ylabel('Genauigkeit [%]')
    plt.grid()
    # plt.savefig('plots/Aufgabe1/mean/mean_dataset' + name + '.eps', format='eps')
    plt.show()


def cv_sets(X_tr, y_tr, n_fold):
    """

    Parameters
    ----------
    X_tr : 2D array

    y_tr : 1D array

    n_fold : int

    Return
    ----------
    cv_set_X_tr : list

    cv_set_y_tr : list
    """

    cv_set_X_tr = []
    cv_set_y_tr = []

    for fold in range(0, n_fold, 1):
        fold_size = len(X_tr) // n_fold

        train_data = X_tr[fold * fold_size: (fold + 1) * fold_size:]
        train_labels = y_tr[fold * fold_size: (fold + 1) * fold_size:]

        cv_set_X_tr.append(train_data)
        cv_set_y_tr.append(train_labels)

    return (cv_set_X_tr, cv_set_y_tr)

def crossvalidation(set_X_cv, set_y_cv, j):
    """

    Parameters
    ----------
    set_X_cv : list

    set_y_cv : list

    j : int

    Return
    ----------
    corr_array : list
    """

    corr_array = []
    for i, test_point in enumerate(set_X_cv):
        X_cv_tr = np.delete(set_X_cv, i, axis=0)
        X_cv_tr = np.vstack(X_cv_tr)
        y_cv_tr = np.delete(set_y_cv, i, axis=0)
        y_cv_tr = np.hstack(y_cv_tr)

        X_cv_te = set_X_cv[i]
        y_cv_te = set_y_cv[i]

        test_p = kNN(X_cv_tr, y_cv_tr, X_cv_te, j)
        corr = validate_test_points(test_p, y_cv_te)
        corr_array.append(corr)

    return (corr_array)


#---------------------------------------------------------------------------------------------------

# Aufgabe 1a)

# Lade files
mk = np.loadtxt('data/perceptrondata.txt')
lb = np.loadtxt('data/perceptrontarget2.txt')

lb = lb.astype(int)

n = 5 # Anzahl Datensets
size_train = 0.5 # Größe der Trainingsmenge
k = list(range(1, 21, 2))

# Für die Reproduzierbarkeit
random.seed(759)
random_seeds = random.sample(range(10000000), n)

results = pd.DataFrame({'k':k})
arr = []
i = 0
for seed in random_seeds:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, size_train, seed)
    arr = []
    for j in k:
        test_p = kNN (X_tr, y_tr, X_te, j)
        corr = validate_test_points(test_p, y_te)
        arr.append(corr)
    results['dataset%s'%i] = arr
    i = i + 1


# Plot
plot_line(results,'0')
plot_mean(results,'0')

#---------------------------------------------------------------------------------------------------
# Aufgabe 1b)

data = {'k': k}
n_folds = [5, 10]

i=0
l=0
for i_seed in random_seeds:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, size_train, i_seed)
    set_X_cv, set_y_cv = cv_sets(X_tr, y_tr, n_folds[0])
    d0 = []
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    for j in k:
        cross=crossvalidation(set_X_cv, set_y_cv, j)
        d0.append(cross[0])
        d1.append(cross[1])
        d2.append(cross[2])
        d3.append(cross[3])
        d4.append(cross[4])
    if l==0:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4])})
        dataset1 = pd.DataFrame(data)
    if l==1:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4])})
        dataset2 = pd.DataFrame(data)
    if l==2:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4])})
        dataset3 = pd.DataFrame(data)
    if l==3:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4])})
        dataset4 = pd.DataFrame(data)
    if l==4:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4])})
        dataset5 = pd.DataFrame(data)
    i = i + 1
    l = l + 1


# Plots
plot_line(dataset1,'1')
plot_mean(dataset1,'1')
plot_line(dataset2,'2')
plot_mean(dataset2,'2')
plot_line(dataset3,'3')
plot_mean(dataset3,'3')
plot_line(dataset4,'4')
plot_mean(dataset4,'4')
plot_line(dataset5,'5')
plot_mean(dataset5,'5')


i=0
l=0
for i_seed in random_seeds:
    X_tr, y_tr, X_te, y_te = build_testtrain(mk, lb, size_train, i_seed)
    set_X_cv, set_y_cv = cv_sets(X_tr, y_tr, n_folds[1])

    d0 = []
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    d6 = []
    d7 = []
    d8 = []
    d9 = []

    for j in k:
        cross=crossvalidation(set_X_cv, set_y_cv, j)
        d0.append(cross[0])
        d1.append(cross[1])
        d2.append(cross[2])
        d3.append(cross[3])
        d4.append(cross[4])
        d5.append(cross[5])
        d6.append(cross[6])
        d7.append(cross[7])
        d8.append(cross[8])
        d9.append(cross[9])
    if l==0:
        #dataset1=pd.DataFrame([k,d0,d1,d2,d3,d4],columns=['k',"dataset0","dataset1","dataset2","dataset3","dataset4"])
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])})
        dataset1 = pd.DataFrame(data)
    if l==1:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])})
        dataset2 = pd.DataFrame(data)
    if l==2:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])})
        dataset3 = pd.DataFrame(data)
    if l==3:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])})
        dataset4 = pd.DataFrame(data)
    if l==4:
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9']
        data.update({dataset: d for dataset, d in zip(datasets, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])})
        dataset5 = pd.DataFrame(data)
    i = i + 1
    l = l + 1


# Plots
plot_line(dataset1,'1_1')
plot_mean(dataset1,'1_1')
plot_line(dataset2,'2_1')
plot_mean(dataset2,'2_1')
plot_line(dataset3,'3_1')
plot_mean(dataset3,'3_1')
plot_line(dataset4,'4_1')
plot_mean(dataset4,'4_1')
plot_line(dataset5,'5_1')
plot_mean(dataset5,'5_1')


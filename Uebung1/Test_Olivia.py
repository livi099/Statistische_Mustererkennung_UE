# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris

# te=test(X_tr, y_tr, X_te, y_te,j)
#
# def test(X_train, y_train, X_test, y_test,n):
#     knn = KNeighborsClassifier(n_neighbors=n)
#
#     knn.fit(X_train, y_train)
#     # Predict on dataset which model has not seen before
#     predict=knn.predict(X_test)
#     score=knn.score(X_test, y_test)

"""
UE Statistische Mustererkennung - WS2022
Gruppe 5
Elisabeth Ötsch - 01425712
Lisa Kern - 01604671
Sukrit Sharma - 01429163

Aufgabe UE-I.1
"""
# load packages
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd


# functions
def euclidean_distance(x_1, x_2):
    # input: two arrays with two elements each
    # output: euclidean distance

    dis = ((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2) ** 0.5
    return dis


"""
Implementieren Sie die Funktion y te = kNN(X tr, y tr, X te, k), welche
– gegeben die Trainingsmenge (Merkmale, Label) = (X tr, y tr ) – mittels
kNN die Klassen-Labels fur die Test-Merkmalsvektoren in X ¨ te berechnet.
Sie k¨onnen annehmen, dass es nur 2 Klassen gibt.

"""


def knn(x_tr, y_tr, x_te, k):
    """ function for kNN
        input:  x_tr - training array with two columns
                y_tr - label for training array with single column
                x_te - test array with two columns
                k - integer number of nearest neighbours

        Returns
        -------
        class of given test points as x_te
    """

    # test input

    # x_tr = data
    # y_tr = target
    # x_te = x_tr[0:5]
    # k=9

    # x_tr = Merkmale[training_data_index]
    # y_tr = Label[training_data_index]
    # x_te = Merkmale[testing_data_index]

    # empty array for euclidean distance
    euc_distance = np.empty((len(x_tr), len(x_te)))

    # store euclidean distance of each testing points with training points as
    # multidimensional array where each column is for each testing points and
    # rows for training points

    for n_te in range(0, len(x_te), 1):
        for n_tr in range(0, len(x_tr), 1):
            euc_distance[n_tr, n_te] = euclidean_distance(x_tr[n_tr], x_te[n_te])

    # classify testing points based on their euclidean distance with training
    # points and labels of training points and store their labels based on kNN
    class_of_test_points = []
    for n_col in range(0, len(x_te)):
        distance_column = euc_distance[:, n_col]
        k_min_index = distance_column.argsort()[:2 * k]
        labels_with_min_distance = y_tr[k_min_index]
        labels = np.unique(labels_with_min_distance)
        if len(labels) == 1:
            selected_label = labels[0]
        else:
            first_label = labels[0]
            second_label = labels[1]
            first_label_boolean = labels_with_min_distance == first_label
            if sum(first_label_boolean) != k:
                if sum(first_label_boolean) > k:
                    selected_label = first_label
                else:
                    selected_label = second_label
            else:
                sum_distance_first_label = sum(first_label_boolean * distance_column[k_min_index])
                sum_distance_second_label = sum((1 - first_label_boolean) * distance_column[k_min_index])
                if sum_distance_second_label > sum_distance_first_label:
                    selected_label = first_label
                else:
                    selected_label = second_label
        class_of_test_points = np.append(class_of_test_points, selected_label)
    return (class_of_test_points)


def plot_k_and_accuracy(k_and_accuracy, type_plot, title_text):
    """

    input : k_and_accuracy: array of classification accuracy for each K with first column as  K value
                            and each of other columns as accuracy for different cases
            type_plot: different case scenario that goes into plot label. type empty string ('') for no inputs
            title_text: Additional text for title . type empty string ('') for no inputs

    Returns
    -------
    Line plot to show the accuracy for each k NN

    """
    fig_k_and_accuracy = plt.figure()
    ax = plt.axes()
    for i in range(1, len(k_and_accuracy[0])):
        ax.plot(k_and_accuracy[:, 0], k_and_accuracy[:, i], linewidth=0.8, linestyle='dashed', label=type_plot + str(i))
    # calculate mean of accuracy for each k
    mean_accuracy = np.mean(k_and_accuracy[:, 1:len(k_and_accuracy[0])], axis=1)
    ax.plot(k_and_accuracy[:, 0], mean_accuracy, linewidth=3, label='Mean Accuracy')
    max_accuracy = max(mean_accuracy)
    max_accuracy_k = k_and_accuracy[:, 0][np.argmax(mean_accuracy)]
    ax.scatter(max_accuracy_k, max_accuracy, marker="X", label="Optimal Accuracy", s=80, color='green')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of kNN' + title_text)
    ax.set_ylim(None, 105)
    ax.set_xlim(0, max(k_and_accuracy[:, 0] - 1))
    plt.xticks(range(1, np.int_(max(k_and_accuracy[:, 0]) + 1), 2))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid(True)
    fig_k_and_accuracy.tight_layout()
    return (fig_k_and_accuracy)


def random_subset_data(i_seed, Merkmale, Label):
    """
    selects different subsets of features and label based on seed value.

    Parameters
    ----------
    i_seed : integer
        seed for constant output.
    Merkmale : 2D array
        2D array of features.
    Label : array
        single domentional array of label.

    Returns
    -------
    training and testing set

    """
    random.seed(i_seed)
    training_data_index = random.sample(range(len(Merkmale)), 4 * np.int_(len(Merkmale) / 5))
    training_data_index = random.sample(range(len(Merkmale)),100)
    #print(len(training_data_index ))
    testing_data_index = [x for x in range(len(Merkmale)) if x not in training_data_index]
    x_tr = Merkmale[training_data_index]
    y_tr = Label[training_data_index]
    x_te = Merkmale[testing_data_index]
    y_te = Label[testing_data_index]
    return [x_tr, y_tr, x_te, y_te]


def compute_accuracy(x_tr, y_tr, x_te, y_te, k_range):
    """


    Parameters
    ----------
    x_tr : 2D array
        array of training set.
    y_tr : array
        array of labels for training set.
    x_te : 2D array
        array of testing set.
    y_te : array
        lables of testing set.
    k_range : array
        array of k as integers.

    Returns
    -------
    accuracy as 2D array

    """

    accuracy_all = []
    for k in k_range:
        y_te_knn = knn(x_tr, y_tr, x_te, k)
        accuracy = (sum(y_te == y_te_knn) / len(y_te) * 100)
        accuracy_all = np.append(accuracy_all, accuracy)

    return (accuracy_all)


def seperate_crossvalidation_set(x_tr, y_tr, n_fold):
    """
    Parameters
    ----------
    x_tr : 2D array
        array of training set.
    yy_tr : array
        array of labels for training set.
    n_fold : int
        number of dataset outpur.

    Returns
    -------
    list
        Seperated n_fold datasets for cross-validation.

    """

    cv_set_x_tr = []
    cv_set_y_tr = []
    for i in range(0, n_fold, 1):
        number_of_rows = len(x_tr) / n_fold
        print('number',number_of_rows)
        print(len(x_tr))
        print(len(y_tr))

        x_tr_i = x_tr[np.int_(i * number_of_rows):np.int_((i + 1) * number_of_rows)]
        y_tr_i = y_tr[np.int_(i * number_of_rows):np.int_((i + 1) * number_of_rows)]
        cv_set_x_tr.append(x_tr_i)
        cv_set_y_tr.append(y_tr_i)

    print(cv_set_x_tr)
    print(cv_set_y_tr)
    return [cv_set_x_tr, cv_set_y_tr]


def df_from_array(input_array):
    """
    takes input array and converts to dataframe with first row as columndead
    """

    df = pd.DataFrame(np.transpose(input_array), columns=input_array[:, 0])
    df = df.iloc[1:, :]
    return df


# output
if __name__ == '__main__':
    # load input files

    Merkmale = np.loadtxt('data/perceptrondata.txt')
    Label = np.loadtxt('data/perceptrontarget2.txt')

    # # combining features and label
    # dataset = np.hstack((Merkmale, np.array([Label]).transpose()))

    # define k range
    k_range = range(1, 21, 2)

    # random seed for constant results
    random.seed(1234)

    # number of input data set repetition as mentioned in task 1 a
    repetation = 5

    # matrix with K as a first column
    k_and_accuracy = np.array([k_range]).transpose()

    # multiple seeds to choose ramdom data from dataset for each repetition
    random_seeds = random.sample(range(10000000), repetation)

    """
    Aufgabe UE-I.1
    Unterteilen Sie die Daten in den Eingabedateien
    (Merkmale, Label) = (perceptrondata.txt perceptrontarget2.txt)
    wiederholt, mindestens jedoch 5 mal, in eine jeweils gleich große Testund 
    Trainingsmenge. Ermitteln Sie Test- und Trainingsfehler fur ver- ¨
    schiedene Werte von k (z.B [1, 3, 5, .., 17 ]), und stellen Sie diese in
    geeigneter Form dar.
    """

    for i_seed in random_seeds:
        x_tr, y_tr, x_te, y_te = random_subset_data(i_seed, Merkmale, Label)
        accuracy_all = compute_accuracy(x_tr, y_tr, x_te, y_te, k_range)
        k_and_accuracy = np.hstack((k_and_accuracy, np.array([accuracy_all]).transpose()))
    plot_k_and_accuracy(k_and_accuracy, 'Dataset:', '').show()
    plt.figure()
    sns.boxplot(data=df_from_array(k_and_accuracy), showmeans=True, whis=[0, 100])
    mean_accuracy = np.mean(k_and_accuracy[:, 1:len(k_and_accuracy[0])], axis=1)
    max_accuracy = max(mean_accuracy)
    max_accuracy_k = ((k_and_accuracy[:, 0][np.argmax(mean_accuracy)]) - 1) / 2
    plt.scatter(max_accuracy_k, max_accuracy, marker="X", label="Optimal Accuracy", s=80, color='blue')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of kNN: Different Datasets')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    # Task 1 b

    """
    2 Aufgabe UE-I.2
    Bezeichne X die geworfene Augenzahl eines fairen, 6-seitigen Wurfels. Be- ¨
    zeichne weiters A das Ereignis X ≥ 4 und B das Ereignis gerade(X).
    a) Berechnen Sie P(A ∪ B) mittels der Summmenregel.
    b) Berechnen Sie die 2 × 2 Kontingenztafel bzg. der obigen Ereignisse.
    Diese l¨asst sich auch als Kontingenztafel zweier abgeleiteter boolscher
    Zufallsvariablen mit Y = X ≥ 4 und Z = gerade(X) auffassen. Sind
    Y und Z unabh¨angig?
    """

    # 5-Fold CV
    for i_seed, i_iteration in zip(random_seeds, range(1, len(random_seeds) + 1)):
        x_tr, y_tr, x_te, y_te = random_subset_data(i_seed, Merkmale, Label)
        cv_set_x, cv_set_y = seperate_crossvalidation_set(x_tr, y_tr, n_fold=5)
        k_and_accuracy_cv = np.array([k_range]).transpose()
        for i in range(0, len(cv_set_x)):

            x_tr_cv = cv_set_x.copy()
            x_tr_cv.pop(i)
            x_tr_cv = np.vstack(x_tr_cv)
            y_tr_cv = cv_set_y.copy()
            y_tr_cv.pop(i)
            y_tr_cv = np.hstack(y_tr_cv)

            x_te_cv = cv_set_x[i]
            y_te_cv = cv_set_y[i]

            accuracy_cv = compute_accuracy(x_tr_cv, y_tr_cv,
                                           x_te_cv, y_te_cv, k_range)
            k_and_accuracy_cv = np.hstack((k_and_accuracy_cv, np.array([accuracy_cv]).transpose()))
        plot_k_and_accuracy(k_and_accuracy_cv, 'CV-Five-Fold:', title_text=': Dataset' + str(i_iteration)).show()
        plt.figure()
        sns.boxplot(data=df_from_array(k_and_accuracy_cv), showmeans=True, whis=[0, 100])
        mean_accuracy = np.mean(k_and_accuracy_cv[:, 1:len(k_and_accuracy_cv[0])], axis=1)
        max_accuracy = max(mean_accuracy)
        max_accuracy_k = ((k_and_accuracy[:, 0][np.argmax(mean_accuracy)]) - 1) / 2
        plt.scatter(max_accuracy_k, max_accuracy, marker="X", label="Optimal Accuracy", s=80, color='blue')
        plt.xlabel('k')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of kNN: 5-fold Crossvalidation')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    # 10-fold crossvalidation
    for i_seed, i_iteration in zip(random_seeds, range(1, len(random_seeds) + 1)):
        x_tr, y_tr, x_te, y_te = random_subset_data(i_seed, Merkmale, Label)
        cv_set_x, cv_set_y = seperate_crossvalidation_set(x_tr, y_tr, n_fold=10)
        k_and_accuracy_cv = np.array([k_range]).transpose()
        for i in range(0, len(cv_set_x)):
            x_tr_cv = cv_set_x.copy()
            x_tr_cv.pop(i)
            x_tr_cv = np.vstack(x_tr_cv)
            y_tr_cv = cv_set_y.copy()
            y_tr_cv.pop(i)
            y_tr_cv = np.hstack(y_tr_cv)
            x_te_cv = cv_set_x[i]
            y_te_cv = cv_set_y[i]
            accuracy_cv = compute_accuracy(x_tr_cv, y_tr_cv,
                                           x_te_cv, y_te_cv, k_range)
            k_and_accuracy_cv = np.hstack((k_and_accuracy_cv, np.array([accuracy_cv]).transpose()))
        plot_k_and_accuracy(k_and_accuracy_cv, 'CV-Ten-Fold:', title_text=': Dataset' + str(i_iteration)).show()
        plt.figure()
        sns.boxplot(data=df_from_array(k_and_accuracy_cv), showmeans=True, whis=[0, 100])
        mean_accuracy = np.mean(k_and_accuracy_cv[:, 1:len(k_and_accuracy_cv[0])], axis=1)
        max_accuracy = max(mean_accuracy)
        max_accuracy_k = ((k_and_accuracy[:, 0][np.argmax(mean_accuracy)]) - 1) / 2
        plt.scatter(max_accuracy_k, max_accuracy, marker="X", label="Optimal Accuracy", s=80, color='blue')
        plt.xlabel('k')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of kNN: 10-fold Crossvalidation')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

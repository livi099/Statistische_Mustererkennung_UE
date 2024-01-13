# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


#Funktionen
def entscheidungsgrenze(Mittelwert, Wahrscheinlichkeit, Kovarianzmatrix, a, b, x):
    w = np.dot(np.linalg.inv(Kovarianzmatrix), np.matrix(Mittelwert[a] - Mittelwert[b]).T)

    temp = np.matrix(Mittelwert[a] - Mittelwert[b])
    temp = temp/(np.dot(np.dot(temp, np.linalg.inv(Kovarianzmatrix)), temp.T))

    b = 1 / 2 * (Mittelwert[a] + Mittelwert[b]) - (temp) * np.log(Wahrscheinlichkeit[a]/Wahrscheinlichkeit[b])

    #grenze = (w[1, 0] * b[0, 1] + w[0, 0] * b[0, 0] - w[0, 0] * x) / w[1, 0]

    return temp




# Import Data
ldaTrain = np.loadtxt('./data/ldaTrain.txt')
ldaTest = np.loadtxt('./data/ldaTest.txt')

# Gesamtanzahl der Einträge
num_data = len(ldaTrain)
num_class = 3

# Aufteilung in Klassen
class1 = ldaTrain[0:200]
class2 = ldaTrain[200:400]
class3 = ldaTrain[400:500]

class_values = [class1, class2, class3]

# Klassenmittel
mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2, axis=0)
mean3 = np.mean(class3, axis=0)

mean_values = [mean1, mean2, mean3]

# A priori Wahrscheinlichkeiten
pw1 = len(class1)/num_data #2/5
pw2 = len(class2)/num_data #2/5
pw3 = len(class3)/num_data #1/5

pw_values = [pw1, pw2, pw3]

# Berechnung der Kovarianzmatrix
C = []
for i in range(num_class):
    temp = class_values[i] - mean_values[i]
    temp = np.dot(temp.T, temp)
    C.append(temp)

fak = 1/(len(ldaTrain)-num_class)
C = fak*sum(C)


x = np.array([-10, 15])

test = entscheidungsgrenze(mean_values, pw_values, C, 0, 1, x)



print(test)





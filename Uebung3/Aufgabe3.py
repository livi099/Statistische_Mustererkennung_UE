# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.optimize import fsolve

#Funktionen
def entscheidungsgrenze(Mittelwert, Wahrscheinlichkeit, Kovarianzmatrix, j, k, x):
    # Berechnung der Entscheidungsgrenze zwischen zwei Klassen j und k
    # Formel 290
    inv_cov = np.linalg.inv(Kovarianzmatrix)
    w = np.dot(inv_cov, np.matrix(Mittelwert[j] - Mittelwert[k]).T)

    # Formel 291
    temp = np.matrix(Mittelwert[j] - Mittelwert[k])
    temp = temp/(np.dot(np.dot(temp, inv_cov), temp.T))

    b = 1 / 2 * (Mittelwert[j] + Mittelwert[k]) - (temp) * np.log(Wahrscheinlichkeit[j]/Wahrscheinlichkeit[k])

    grenze = (w[1, 0] * b[0, 1] + w[0, 0] * b[0, 0] - w[0, 0] * x) / w[1, 0]

    return grenze

def schnittpunkt(grenze1, grenze2,x):
    # Berechnung des Schnittpunktes zweier Entscheidungsgrenzen
    polyfit_f1 = np.polyfit(x, grenze1, 1)
    polyfit_f2 = np.polyfit(x, grenze2, 1)

    poly1d_f1 = np.poly1d(polyfit_f1)
    poly1d_f2 = np.poly1d(polyfit_f2)

    x_wert = fsolve(lambda x: poly1d_f1(x) - poly1d_f2(x), 0.0)
    y_wert = poly1d_f1(x_wert)

    return x_wert.item(), y_wert.item()

def fehlerberechnung(Testwert, Klassengrenze):
    # Prüfen, ob klassifizierter Werte in richtiger Klasse ist
    border = path.Path(Klassengrenze)
    inlier = border.contains_points(Testwert)

    # Summe aller falsch klassifizierten Werte
    fehler = 0
    for i in range(0, len(inlier)):
        if not inlier[i]:
            fehler = fehler + 1

    return fehler

# Import Data
data_train = np.loadtxt('./data/ldaTrain.txt')
data_test = np.loadtxt('./data/ldaTest.txt')

# Gesamtanzahl der Einträge
num_data = len(data_train)
num_class = 3

# Trainingsdaten
train1 = data_train[0:200]
train2 = data_train[200:400]
train3 = data_train[400:500]

train_values = [train1, train2, train3]

# Testdaten
test1 = data_test[0:200]
test2 = data_test[200:400]
test3 = data_test[400:500]

test_values = [test1, test2, test3]

# Klassenmittel
mean_values = [np.mean(train_values, axis=0) for train_values in train_values]

# A priori Wahrscheinlichkeiten
pw_values = [len(train_values) for train_values in train_values]
pw_values = [pw_values/num_data for pw_values in pw_values]

# Berechnung der Kovarianzmatrix
C = []
for i in range(num_class):
    temp = train_values[i] - mean_values[i]
    temp = np.dot(temp.T, temp)
    C.append(temp)

fak = 1/(len(data_train)-num_class)
C = fak*sum(C)

print("Gepoolte Kovarianzmatrix C:")
for row in C:
    print(" ".join(map(str, row)))

print()

# Definition des Wertebereichs
x = np.array([-10, 15])

# Berechnung der Entscheidungsgrenze
grenze12 = entscheidungsgrenze(mean_values, pw_values, C, 0, 1, x)
grenze13 = entscheidungsgrenze(mean_values, pw_values, C, 0, 2, x)
grenze23 = entscheidungsgrenze(mean_values, pw_values, C, 1, 2, x)

# Berechnung des Schnittpunktes
schnitt_x, schnitt_y = schnittpunkt(grenze12, grenze13,x)

# Führen zum Gleichen Ergebnis, daher auskommentiert
# schnitt_x, schnitt_y = schnittpunkt(grenze12, grenze23,x)
# schnitt_x, schnitt_y = schnittpunkt(grenze13, grenze23,x)

# Plot
# Trainingsdaten
plt.figure(figsize=(6, 5))
plt.scatter(train_values[0][:,0], train_values[0][:,1], 10, edgecolor='blue', facecolors='none', label='\u03C9\u2081')
plt.scatter(train_values[1][:,0], train_values[1][:,1], 10, edgecolor='green', facecolors='none', label='\u03C9\u2082')
plt.scatter(train_values[2][:,0], train_values[2][:,1], 10, edgecolor='orange', facecolors='none', label='\u03C9\u2083')
# Klassengrenzen
plt.plot([schnitt_x, 15],[schnitt_y, grenze12[1]], linestyle='dashed', color='black', label='Klassengrenze',linewidth=0.5)
plt.plot([-10, schnitt_x],[grenze13[0], schnitt_y], linestyle='dashed', color='black', linewidth=0.5)
plt.plot([schnitt_x, 15],[schnitt_y, grenze23[1]], linestyle='dashed', color='black', linewidth=0.5)
# Beschriftung
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim((-17,8))
plt.xlim((-10,15))
plt.title('Trainingsdaten mit Klassengrenzen')
plt.legend(loc='lower right')
plt.savefig('plots/Aufgabe3/Trainingsdaten.eps', format='eps')
plt.show()

# Testdaten
plt.figure(figsize=(6, 5))
plt.scatter(test_values[0][:,0], test_values[0][:,1], 10, edgecolor='blue', facecolors='none', label='\u03C9\u2081')
plt.scatter(test_values[1][:,0], test_values[1][:,1], 10, edgecolor='green', facecolors='none', label='\u03C9\u2082')
plt.scatter(test_values[2][:,0], test_values[2][:,1], 10, edgecolor='orange', facecolors='none', label='\u03C9\u2083')
# Klassengrenzen
plt.plot([schnitt_x, 15],[schnitt_y, grenze12[1]], linestyle='dashed', color='black', label='Klassengrenze',linewidth=0.5)
plt.plot([-10, schnitt_x],[grenze13[0], schnitt_y], linestyle='dashed', color='black',linewidth=0.5)
plt.plot([schnitt_x, 15],[schnitt_y, grenze23[1]], linestyle='dashed', color='black',linewidth=0.5)
# Beschriftung
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim((-17,8))
plt.xlim((-10,15))
plt.title('Testdaten mit Klassengrenzen')
plt.legend(loc='lower right')
plt.savefig('plots/Aufgabe3/Testdaten.eps', format='eps')
plt.show()

# Berechnung der falsch klassifizierten Werte
# Klassengrenzen
Klassengrenze_1 = [(x[0], grenze13[0]), (schnitt_x, schnitt_y), (x[1], grenze12[1]), (x[0], grenze12[1])]
Klassengrenze_2 = [(x[1], grenze23[1]), (schnitt_x, schnitt_y), (x[1], grenze12[1])]
Klassengrenze_3 = [(x[0], grenze13[0]), (schnitt_x, schnitt_y), (x[1], grenze23[1]), (x[0], grenze23[1])]

# Trainingsdaten
abs_train_1 = fehlerberechnung(train_values[0], Klassengrenze_1)
rel_train_1 = 100/len(train1)*abs_train_1
abs_train_2 = fehlerberechnung(train_values[1], Klassengrenze_2)
rel_train_2 = 100/len(train2)*abs_train_2
abs_train_3 = fehlerberechnung(train_values[2], Klassengrenze_3)
rel_train_3 = 100/len(train3)*abs_train_3

# Testdaten
abs_test_1 = fehlerberechnung(test_values[0], Klassengrenze_1)
rel_test_1 = 100/len(test1)*abs_test_1
abs_test_2 = fehlerberechnung(test_values[1], Klassengrenze_2)
rel_test_2 = 100/len(test2)*abs_test_2
abs_test_3 = fehlerberechnung(test_values[2], Klassengrenze_3)
rel_test_3 = 100/len(test3)*abs_test_3

# Gesamtfehler
abs_train_all = abs_train_1 + abs_train_2 + abs_train_3
abs_test_all = abs_test_1 + abs_test_2 + abs_test_3
rel_train_all = 100/len(data_train)*abs_train_all
rel_test_all = 100/len(data_test)*abs_test_all

# Ausgabe
print(f"Absoluter Fehler der klassifizierten Trainingsdaten:")
print(f"Klasse \u03C9\u2081: {abs_train_1}")
print(f"Klasse \u03C9\u2082: {abs_train_2}")
print(f"Klasse \u03C9\u2083: {abs_train_3}")
print()
print(f"Relativer Fehler der klassifizierten Trainingsdaten:")
print(f"Klasse \u03C9\u2081: {rel_train_1}%")
print(f"Klasse \u03C9\u2082: {rel_train_2}%")
print(f"Klasse \u03C9\u2083: {rel_train_3}%")
print()
print(f"Absoluter Fehler der klassifizierten Testdaten:")
print(f"Klasse \u03C9\u2081: {abs_test_1}")
print(f"Klasse \u03C9\u2082: {abs_test_2}")
print(f"Klasse \u03C9\u2083: {abs_test_3}")
print()
print(f"Relativer Fehler der klassifizierten Testdaten:")
print(f"Klasse \u03C9\u2081: {rel_test_1}%")
print(f"Klasse \u03C9\u2082: {rel_test_2}%")
print(f"Klasse \u03C9\u2083: {rel_test_3}%")
print()
print(f"Absoluter Fehler aller Trainingswerte: {abs_train_all}")
print(f"Relativer Fehler aller Trainingswerte: {rel_train_all}%")
print(f"Absoluter Fehler aller Testwert: {abs_test_all}")
print(f"Relativer Fehler aller Testwert: {rel_test_all}%")
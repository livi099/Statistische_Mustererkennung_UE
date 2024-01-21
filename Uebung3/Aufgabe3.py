# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.optimize import fsolve

#Funktionen
def entscheidungsgrenze(Mittelwert, Wahrscheinlichkeit, Kovarianzmatrix, j, k, x):
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
    f1 = np.polyfit(x, grenze1, 1)
    f2 = np.polyfit(x, grenze2, 1)

    poly1d_f1 = np.poly1d(f1)
    poly1d_f2 = np.poly1d(f2)

    x_wert = fsolve(lambda x: poly1d_f1(x) - poly1d_f2(x), 0.0)
    y_wert = poly1d_f1(x_wert)

    return x_wert.item(), y_wert.item()

def fehlerberechnung(Testwert, p):
    # Berechnung, ob Wert in Klasse liegt
    inlier = p.contains_points(Testwert)

    fehler = 0
    for i in range(0, len(inlier)):
        if not inlier[i]:
            fehler = fehler + 1

    return fehler

# Import Data
ldaTrain = np.loadtxt('./data/ldaTrain.txt')
ldaTest = np.loadtxt('./data/ldaTest.txt')

# Gesamtanzahl der Einträge
num_data = len(ldaTrain)
num_class = 3

# Trainingsdaten
train1 = ldaTrain[0:200]
train2 = ldaTrain[200:400]
train3 = ldaTrain[400:500]

train_values = [train1, train2, train3]

# Testdaten
test1 = ldaTest[0:200]
test2 = ldaTest[200:400]
test3 = ldaTest[400:500]

test_values = [test1, test2, test3]

# Klassenmittel
mean1 = np.mean(train1, axis=0)
mean2 = np.mean(train2, axis=0)
mean3 = np.mean(train3, axis=0)

mean_values = [mean1, mean2, mean3]

# A priori Wahrscheinlichkeiten
pw1 = len(train1)/num_data #2/5
pw2 = len(train2)/num_data #2/5
pw3 = len(train3)/num_data #1/5

pw_values = [pw1, pw2, pw3]

# Berechnung der Kovarianzmatrix
C = []
for i in range(num_class):
    temp = train_values[i] - mean_values[i]
    temp = np.dot(temp.T, temp)
    C.append(temp)

fak = 1/(len(ldaTrain)-num_class)
C = fak*sum(C)

x = np.array([-10, 15])

grenze1 = entscheidungsgrenze(mean_values, pw_values, C, 0, 1, x)
grenze2 = entscheidungsgrenze(mean_values, pw_values, C, 0, 2, x)
grenze3 = entscheidungsgrenze(mean_values, pw_values, C, 1, 2, x)

# Berechnung des Schnittpunktes
schnitt_x, schnitt_y = schnittpunkt(grenze1, grenze2,x)

#Plot
# Trainingsdaten
plt.figure(figsize=(6, 5))
plt.scatter(train_values[0][:,0], train_values[0][:,1], 20, edgecolor='#879e82', facecolors='none', label='Klasse 1')
plt.scatter(train_values[1][:,0], train_values[1][:,1], 20, edgecolor='#194a7a', facecolors='none', label='Klasse 2')
plt.scatter(train_values[2][:,0], train_values[2][:,1], 20, edgecolor='#c7522a', facecolors='none', label='Klasse 3')

plt.plot([schnitt_x, 15],[schnitt_y, grenze1[1]],'k-.', label='Klassengrenze',linewidth=0.8)
plt.plot([-10, schnitt_x],[grenze2[0], schnitt_y],'k-.',linewidth=0.8)
plt.plot([schnitt_x, 15],[schnitt_y, grenze3[1]],'k-.',linewidth=0.8)

plt.xlabel('X')
plt.ylabel('Y')
plt.ylim((-17,8))
plt.xlim((-10,15))
plt.title('Trainingsdaten und Klassengrenzen')
plt.legend(loc='lower right')
plt.show()

# Testdaten
plt.figure(figsize=(6, 5))
plt.scatter(test_values[0][:,0], test_values[0][:,1], 20, edgecolor='#879e82', facecolors='none', label='Klasse 1')
plt.scatter(test_values[1][:,0], test_values[1][:,1], 20, edgecolor='#194a7a', facecolors='none', label='Klasse 2')
plt.scatter(test_values[2][:,0], test_values[2][:,1], 20, edgecolor='#c7522a', facecolors='none', label='Klasse 3')

plt.plot([schnitt_x, 15],[schnitt_y, grenze1[1]],'k-.', label='Klassengrenze',linewidth=0.8)
plt.plot([-10, schnitt_x],[grenze2[0], schnitt_y],'k-.',linewidth=0.8)
plt.plot([schnitt_x, 15],[schnitt_y, grenze3[1]],'k-.',linewidth=0.8)

plt.xlabel('X')
plt.ylabel('Y')
plt.ylim((-17,8))
plt.xlim((-10,15))
plt.title('Testdaten und Klassengrenzen')
plt.legend(loc='lower right')
plt.show()


# Berechnung der falsch klassifizierten Werte

# Klasse 1
p1 = path.Path([(x[0], grenze2[0]), (schnitt_x, schnitt_y), (x[1], grenze1[1]), (x[0], grenze1[1])])
abs_train_1 = fehlerberechnung(train_values[0], p1)
abs_test_1 = fehlerberechnung(test_values[0], p1)
rel_train_1 = 100/len(train1)*abs_train_1
rel_test_1 = 100/len(test1)*abs_test_1

print(f"Absoluter Fehler der klassifizierten Trainingsdaten (Klasse 1): {abs_train_1}")
print(f"Relativer Fehler der klassifizierten Trainingsdaten (Klasse 1): {rel_train_1}%")
print(f"Absoluter Fehler der klassifizierten Testdaten (Klasse 1): {abs_test_1}")
print(f"Relativer Fehler der klassifizierten Testdaten (Klasse 1): {rel_test_1}%")
print()

# Klasse 2
p2 = path.Path([(x[1], grenze3[1]), (schnitt_x, schnitt_y), (x[1], grenze1[1])])
abs_train_2 = fehlerberechnung(train_values[1], p2)
abs_test_2 = fehlerberechnung(test_values[1], p2)
rel_train_2 = 100/len(train2)*abs_train_2
rel_test_2 = 100/len(test2)*abs_test_2

print(f"Absoluter Fehler der klassifizierten Trainingsdaten (Klasse 2): {abs_train_2}")
print(f"Relativer Fehler der klassifizierten Trainingsdaten (Klasse 2): {rel_train_2}%")
print(f"Absoluter Fehler der klassifizierten Testdaten (Klasse 2): {abs_test_2}")
print(f"Relativer Fehler der klassifizierten Testdaten (Klasse 2): {rel_test_2}%")
print()

# Klasse 3
p3 = path.Path([(x[0], grenze2[0]), (schnitt_x, schnitt_y), (x[1], grenze3[1]), (x[0], grenze3[1])])
abs_train_3 = fehlerberechnung(train_values[2], p3)
abs_test_3 = fehlerberechnung(test_values[2], p3)
rel_train_3 = 100/len(train3)*abs_train_3
rel_test_3 = 100/len(test3)*abs_test_3

print(f"Absoluter Fehler der klassifizierten Trainingsdaten (Klasse 3): {abs_train_3}")
print(f"Relativer Fehler der klassifizierten Trainingsdaten (Klasse 3): {rel_train_3}%")
print(f"Absoluter Fehler der klassifizierten Testdaten (Klasse 3): {abs_test_3}")
print(f"Relativer Fehler der klassifizierten Testdaten (Klasse 3): {rel_test_3}%")
print()

# Gesamtfehler
abs_train_all = abs_train_1 + abs_train_2 + abs_train_3
abs_test_all = abs_test_1 + abs_test_2 + abs_test_3
rel_train_all = 100/len(ldaTrain)*abs_train_all
rel_test_all = 100/len(ldaTest)*abs_test_all

print(f"Absoluter Fehler aller Trainingswerte: {abs_train_all}")
print(f"Relativer Fehler aller Trainingswerte: {rel_train_all}%")
print(f"Absoluter Fehler aller Testwert: {abs_test_all}")
print(f"Relativer Fehler aller Testwert: {rel_test_all}%")
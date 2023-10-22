# Statistische Mustererkennung UE

## Aufgabe 1
Implementieren Sie die Funktion y te = kNN(X tr, y tr, X te, k), welche - gegeben die Trainingsmenge (Merkmale, Label) = (X tr, y tr ) - mittels
kNN die Klassen-Labels für die Test-Merkmalsvektoren in X te berechnet. Sie können annehmen, dass es nur 2 Klassen gibt.

a) Unterteilen Sie die Daten in den Eingabedateien (Merkmale, Label) = (perceptrondata.txt perceptrontarget2.txt)
wiederholt, mindestens jedoch 5 mal, in eine jeweils gleich große Test- und Trainingsmenge. Ermitteln Sie Test- und Trainingsfehler für verschiedene
Werte von k (z.B [1, 3, 5, .., 17 ]), und stellen Sie diese in geeigneter Form dar.

b) Bestimmen Sie füur jede der obigen Aufteilungen in Test- und Trainingsmenge den optimalen Wert für k mittels 5- und 10-facher Kreuzvalidierung.


## Aufgabe 2
Bezeichne X die geworfene Augenzahl eines fairen, 6-seitigen Würfels. Bezeichne weiters A das Ereignis X > 4 und B das Ereignis gerade(X).

a) Berechnen Sie P(A U B) mittels der Summmenregel.

b) Berechnen Sie die 2 x 2 Kontingenztafel bzg. der obigen Ereignisse. 
Diese lässt sich auch als Kontingenztafel zweier abgeleiteter boolscher Zufallsvariablen mit Y = X > 4 und Z = gerade(X) auffassen. Sind
Y und Z unabhängig?
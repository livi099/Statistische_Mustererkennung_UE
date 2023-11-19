# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2

# Funktionen
def df(x, my, sigma):
    """
    Berechnung der Dichtefunktion (Normalverteilung)

    Parameters
    ----------
    x : list (Merkmale)
    my : double (Mittelwert)
    sigma : double (Standardabweichung)

    Return
    ----------
    p_x : list

    """

    p_x = []
    for x_i in x:
        temp1 = -(((x_i-my)**2)/(2*(sigma**2)))
        temp2 = (1/(((2*np.pi)**(1/2))*sigma))*np.exp(temp1)
        p_x.append(temp2)
    return p_x

# Daten

# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2

# Funktionen
def dichtefunktion(x, mu, sigma):
    """
    Berechnung der Dichtefunktion (Normalverteilung)

    Parameters
    ----------
    x: Merkmale (list)
    mu: Mittelwert (double)
    sigma: Standardabweichung (double)

    Return
    ----------
    df: Dichteverteilung (list)

    """
    df = []
    for x_i in x:
        temp1 = -(((x_i-mu)**2)/(2*(sigma**2)))
        temp2 = (1/(((2*np.pi)**(1/2))*sigma))*np.exp(temp1)
        df.append(temp2)
    return(df)

def randverteilung(c,  p_x_omega1, p_x_omega2, p_omega1, p_omega2):
    """
    Berechnung der Radverteilung (evidence) des Merkmals

    Parameters
    ----------
    c: Klassen (list)
    p_x_omega1: Dichteverteilung (list)
    p_x_omega2: Dichteverteilung (list)
    p_omega1: prior (double)
    p_omega2: prior (double)

    Return
    ----------
    evidence: Randverteilung (list)

    """
    evidence = []
    for i in range(len(c)):
        temp1 = p_x_omega1[i] * p_omega1 + p_x_omega2[i] * p_omega2
        evidence.append(temp1)
    return(evidence)

def posterior(c, p_omega, p_x_omega, p_x):
    """
    Berechnung des posteriors

    Parameters
    ----------
    c: Klassen (list)
    p_omega: prior (double)
    p_x_omega: Dichteverteilung (list)
    p_x: evidence (list)

    Return
    ----------
    p_x : posterior (list)

    """
    p_posterior = []
    for i in range(len(c)):
        temp1 = (p_x_omega[i] * p_omega)/p_x[i]
        p_posterior.append(temp1)
    return(p_posterior)

# Daten
x = np.arange(-20, 15, 0.01)

p_omega1, p_omega2 = 0.3, 0.7 #a priori Wahrscheinlichkeit
mu1, mu2 = -7, 4
sigma1, sigma2 = np.sqrt(30), np.sqrt(13)

# Berechnung der Dichtefunktion für beide Klassen
p_x_omega1 = dichtefunktion(x,mu1, sigma1)
p_x_omega2 = dichtefunktion(x,mu2, sigma2)

# Berechnung der Randverteilung (evidence)
p_x = randverteilung(x,p_x_omega1, p_x_omega2, p_omega1, p_omega2)

# Berechnung des posterior
p_pos1 = posterior(x, p_omega1, p_x_omega1, p_x)
p_pos2 = posterior(x, p_omega2, p_x_omega2, p_x)
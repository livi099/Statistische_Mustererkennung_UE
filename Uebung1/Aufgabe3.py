# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import math as m

# Aufgabe 3
# Binomial Koeffizient

# Die Funktion binom berechnet den Binomailkoeffizient. Ihr müssen die Parameter n und theta uebergeben werden.
# r ist ein Vektor und geht von 0:n (natürlich in python bis n+1)
# Der Vektor results_binom gibt dann alle P(r)-Werte als Vektor aus

def binom(n,theta):
    results_binom = []

    for r in range(n+1):
        p = ((m.factorial(n))/(m.factorial(r)*m.factorial(n-r))) * (theta ** r) * ((1 - theta) ** (n - r))
        results_binom.append(p)

    return(results_binom)


value_binom = binom(3, 0.7)

print(value_binom)


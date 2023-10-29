# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import math as m

# Aufgabe 3
# Binomial Koeffizient

def binom(n,theta):
    results = []

    for r in range(n+1):
        P = ((m.factorial(n))/(m.factorial(r)*m.factorial(n-r))) * (theta ** r) * ((1 - theta) ** (n - r))
        results.append(P)

    return(results)


test = binom(3, 0.7)

print(test)


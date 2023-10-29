# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import scipy.integrate as spi
from scipy.stats import pareto as par
import matplotlib.pyplot as plt


# Aufgabe 4

def pareto_df(x, alpha, x_min):
    """ Differentialgleichung für Pareto-Verteilung

       Parameters
       ----------
       x : float

       alpha : float

       x_min : float

       Return
       ----------
       df : float
       """
    df = alpha * (x_min ** alpha) / (x ** (alpha + 1))

    return df

def pareto_ew(alpha, x_min, n):
    """ Berechnet für die Pareto-Verteilung den (laufenden) Erwartungswert

       Parameters
       ----------

       alpha : float

       x_min : float

       n : 1D array or list

       Return
       ----------
       ew : list
       """
    ew = []
    for i in n:
        if i == np.inf:
            # Integration bis unendlich
            result, _ = spi.quad(lambda x: x * pareto_df(x, alpha, x_min), x_min, np.inf)
        else:
            # Integration bis zu einem endlichen Wert
            result, _ = spi.quad(lambda x: x * pareto_df(x, alpha, x_min), x_min, i)
        ew.append(result)

    return ew


# Eingabeparameter
alpha = [1.16]
x_min = 1.0
n = np.linspace(3, 100, 100)
q = np.linspace(0.0, 1.0, 1000)

# Lorenz Berechnung und Plot
for i in alpha:
    xq = par.ppf(q, i)
    lorenz = np.array(pareto_ew(i, x_min, xq))/float(pareto_ew(i, x_min, [np.inf])[0])

    plt.plot(q, lorenz, label=r'$\alpha$ = ' + str(i))
plt.scatter(0.8, 0.2, s=50, marker= '*', color ='red', label='Pareto-Prinzip', zorder=2)
plt.title(r'Lorenz-Kurve ($x_{min}=1$)')
plt.xlabel('q')
plt.ylabel('L(q)')
plt.legend()
plt.show()


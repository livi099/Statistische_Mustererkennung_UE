# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Aufgabe3

def dichte(x, mu, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.power(x - mu, 2) / (2 * np.power(std, 2)))

def rand(x, p_x_mu, p_mu, p_x_mu2, p_mu2):
    return [p_x_mu[i] * p_mu + p_x_mu2[i] * p_mu2 for i in range(len(x))]

def posterior(x, p_x_mu, p_mu, p_x):
    return [(p_mu[i] * p_x_mu) / p_x[i] for i in range(len(x))]

def intersection_point(pos_H0, pos_H1, x):
    idx = np.argmin(np.abs(np.array(pos_H0) - np.array(pos_H1)))
    point = x[idx]
    return point

# Werte
p_H0 = [0.9, 0.99]
p_H1 = [0.1, 0.01]
mu_H0, mu_H1 = 4, 5
std_H0, std_H1 = np.sqrt(1), np.sqrt(1)

# Aufgabe a)
x_values = np.linspace(0, 12, 1000)

False_P = 1 - stats.norm.cdf(x_values, loc=mu_H0, scale=std_H0)  # false positive
True_P = 1 - stats.norm.cdf(x_values, loc=mu_H1, scale=std_H1)  # true positive

# Vorhersagewert
VW1 = [(True_P[i] * p_H1[0]) / (True_P[i] * p_H1[0] + False_P[i] * p_H0[0]) for i in range(len(x_values))]
VW2 = [(True_P[i] * p_H1[1]) / (True_P[i] * p_H1[1] + False_P[i] * p_H0[1]) for i in range(len(x_values))]

plt.figure(figsize=(18, 12))
plt.plot(x_values, VW1, color='orange', label=r'$H_0=0.9, H_1=0.1$')
plt.plot(x_values, VW2, color='blue', label=r'$H_0=0.99, H_1=0.01$')
plt.title('Positiver Vorhersagewert als Funktion der Entscheidungsgrenze', fontsize=24)
plt.ylabel(r'$p(H_1|+)$', fontsize=20)
plt.xlabel(r'$x^*$', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig('plots/Aufgabe3/a.png', format='png')
plt.show()

# Aufgabe b)
px_H0 = dichte(x_values, mu_H0, std_H0)
px_H1 = dichte(x_values, mu_H1, std_H1)
px_9 = rand(x_values, px_H0, p_H0[0], px_H1, p_H1[0])
px_99 = rand(x_values, px_H0, p_H0[1], px_H1, p_H1[1])
p_H0_x_9 = posterior(x_values, p_H0[0], px_H0, px_9)
p_H1_x_9 = posterior(x_values, p_H1[0], px_H1, px_9)
p_H0_x_99 = posterior(x_values, p_H0[1], px_H0, px_99)
p_H1_x_99 = posterior(x_values, p_H1[1], px_H1, px_99)

# Berechnung des Schnittpunkts
ent_grenze1 = intersection_point(p_H0_x_9, p_H1_x_9, x_values)
ent_grenze2 = intersection_point(p_H0_x_99, p_H1_x_99, x_values)

print(f"Die Entscheidungsgrenze bei P(H\u2080)=0.90 und P(H\u2081)=0.10 liegt bei: {round(ent_grenze1, 1)}")
print(f"Die Entscheidungsgrenze bei P(H\u2080)=0.99 und P(H\u2081)=0.01 liegt bei: {round(ent_grenze2, 1)}")

# Berechnung der Bayes Fehlerraten für Signifikanzniveaus 0.9 und 0.99
# Fehlerwahrscheinlichkeiten für Signifikanzniveau 0.9
false_positive_09 = 1 - stats.norm.cdf(ent_grenze1, loc=mu_H0, scale=std_H0) * p_H0[0] #false positiv
false_negative_09 = stats.norm.cdf(ent_grenze1, loc=mu_H1, scale=std_H1) * p_H1[0] #false negativ

# Fehlerwahrscheinlichkeiten für Signifikanzniveau 0.01
true_negative_001 = 1 - stats.norm.cdf(ent_grenze2, loc=mu_H0, scale=std_H0) * p_H0[1] #true negativ
true_positive_001 = stats.norm.cdf(ent_grenze2, loc=mu_H1, scale=std_H1) * p_H1[1]  #true positive

# Berechnung der Bayes Fehlerraten
bayes_error_rate_09 = false_positive_09 + false_negative_09
bayes_error_rate_001 = true_negative_001 + true_positive_001

print(f"Die Bayes-Fehlerrate für P(H\u2080)=0.90 und P(H\u2081)=0.10 liegt bei: {round(bayes_error_rate_09, 2)}")
print(f"Die Bayes-Fehlerrate für P(H\u2080)=0.99 und P(H\u2081)=0.01 liegt bei: {round(bayes_error_rate_001, 2)}")

# Wahrscheinlichkeitsfuntion und Posterioris (H0 = 0.9, H1 = 0.1)
plt.figure(figsize=(18, 12))
p_x_H0 = dichte(x_values, mu_H0, std_H0)
p_x_H1 = dichte(x_values, mu_H1, std_H1)
plt.plot(x_values, p_x_H0, label=r'DF von $H_0$', color='blue', linestyle='--')
plt.plot(x_values, p_x_H1, label=r'DF von $H_1$', color='red', linestyle='--')
plt.plot(x_values, p_H0_x_9, label=r'Posterioris von $H_0$', color='blue')
plt.plot(x_values, p_H1_x_9, label=r'Posterioris von $H_1$', color='red')
plt.plot([ent_grenze1, ent_grenze1], [0, 1], label='Entscheidungsgrenze', linestyle='--', color='k')
plt.title(r'Wahrscheinlichkeitsfuntion und Posterioris von $H_0=0.9$ und $H_1=0.1$', fontsize=24)
plt.grid(True)
plt.ylabel(r'$p(x|H_i)$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('plots/Aufgabe3/b_1.png', format='png')
plt.show()

# Wahrscheinlichkeitsfuntion und Posterioris (H0 = 0.99, H1 = 0.01)
plt.figure(figsize=(18, 12))
p_x_H0 = dichte(x_values, mu_H0, std_H0)
p_x_H1 = dichte(x_values, mu_H1, std_H1)
plt.plot(x_values, p_x_H0, label=r'DF von $H_0$', color='blue', linestyle='--')
plt.plot(x_values, p_x_H1, label=r'DF von $H_1$ ', color='red', linestyle='--')
plt.plot(x_values, p_H0_x_99, label=r'Posterioris von $H_0$', color='blue')
plt.plot(x_values, p_H1_x_99, label=r'Posterioris von $H_1$', color='red')
plt.plot([ent_grenze2, ent_grenze2], [0, 1], label='Entscheidungsgrenze', linestyle='--', color='k')
plt.title(r'Wahrscheinlichkeitsfuntion und Posterioris von $H_0=0.99$ und $H_1=0.01$', fontsize=24)
plt.grid(True)
plt.ylabel(r'$p(x|H_i)$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('plots/Aufgabe3/b_2.png', format='png')
plt.show()

# Plot beider Posterioris und Entscheidungsgrenzen
plt.figure(figsize=(18, 12))
plt.plot(x_values, p_H0_x_9, label=r'$H_0=0.9, H_1=0.1$', color='orange')
plt.plot(x_values, p_H1_x_9, color='orange')
plt.plot(x_values, p_H0_x_99, color='blue')
plt.plot(x_values, p_H1_x_99, label=r'$H_0=0.99, H_1=0.01$', color='blue')
plt.plot([6.7, 6.7], [0, 1], linestyle='--', color='k')
plt.plot([9.1, 9.1], [0, 1], label='Entscheidungsgrenzen', linestyle='--', color='k')
plt.title('Posteriors beider Prävalenzpaare mit Entscheidungsgrenzen', fontsize=24)
plt.grid(True)
plt.ylabel(r'$p(H_i|x)$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.legend(fontsize=20)
idx1 = np.argwhere(np.diff(np.sign(np.array(p_H0_x_9) - np.array(p_H1_x_9)))).flatten()
idx2 = np.argwhere(np.diff(np.sign(np.array(p_H0_x_99) - np.array(p_H1_x_99)))).flatten()
plt.savefig('plots/Aufgabe3/b_3.png', format='png')
plt.show()

# Aufgabe c)
# Plot der ROC Kurve
plt.figure(figsize=(18, 12))
plt.plot(False_P, True_P, color='green')
plt.title('ROC-Kruve (receiver operating characteristic)', fontsize=24)
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'$1 - \beta$', fontsize=20)
plt.grid(True)
plt.savefig('plots/Aufgabe3/c.png', format='png')
plt.show()



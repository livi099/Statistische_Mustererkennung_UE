# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Functions
def density(x, mu, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.power(x - mu, 2) / (2 * np.power(std, 2)))

def likelihood_ratio(x, mu1, std1, mu2, std2, prior1, prior2):
    px1 = density(x, mu1, std1)
    px2 = density(x, mu2, std2)
    return (px2 * prior2) / (px1 * prior1)

def calculate_intersection_point(pos_H0, pos_H1, x):
    idx = np.argmin(np.abs(np.array(pos_H0) - np.array(pos_H1)))
    return x[idx]

def calculate_error_rates(ent_grenze, mu_H0, std_H0, mu_H1, std_H1, p_H0, p_H1):
    false_positive = 1 - stats.norm.cdf(ent_grenze, loc=mu_H0, scale=std_H0) * p_H0
    false_negative = stats.norm.cdf(ent_grenze, loc=mu_H1, scale=std_H1) * p_H1
    return false_positive, false_negative

def plot_probability_functions(x, p_x_H0, p_x_H1, p_H0_x, p_H1_x, decision_boundary, title, filename):
    plt.figure(figsize=(18, 12))
    plt.plot(x, p_x_H0, label=r'DF von $H_0$', color='blue', linestyle='--')
    plt.plot(x, p_x_H1, label=r'DF von $H_1$', color='red', linestyle='--')
    plt.plot(x, p_H0_x, label=r'Posterioris von $H_0$', color='blue')
    plt.plot(x, p_H1_x, label=r'Posterioris von $H_1$', color='red')
    plt.plot([decision_boundary, decision_boundary], [0, 1], label='Entscheidungsgrenze', linestyle='--', color='k')
    plt.title(title, fontsize=24)
    plt.grid(True)
    plt.ylabel(r'$p(x|H_i)$', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(filename, format='png')
    plt.show()

# Werte
p_H0 = np.array([0.9, 0.99])
p_H1 = np.array([0.1, 0.01])
mu_H0, mu_H1 = 4, 5
std_H0, std_H1 = np.sqrt(1), np.sqrt(1)

# Aufgabe a)
x_values = np.linspace(0, 12, 1000)

# Vorhersagewert
VW1 = likelihood_ratio(x_values, mu_H0, std_H0, mu_H1, std_H1, p_H0[0], p_H1[0])
VW2 = likelihood_ratio(x_values, mu_H0, std_H0, mu_H1, std_H1, p_H0[1], p_H1[1])

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
px_H0 = density(x_values, mu_H0, std_H0)
px_H1 = density(x_values, mu_H1, std_H1)
px_9 = likelihood_ratio(x_values, mu_H0, std_H0, mu_H1, std_H1, p_H0[0], p_H1[0])
px_99 = likelihood_ratio(x_values, mu_H0, std_H0, mu_H1, std_H1, p_H0[1], p_H1[1])
p_H0_x_9 = px_9 / (px_9 + p_H0[0])
p_H1_x_9 = 1 - p_H0_x_9
p_H0_x_99 = px_99 / (px_99 + p_H0[1])
p_H1_x_99 = 1 - p_H0_x_99

# Berechnung des Schnittpunkts
ent_grenze1 = calculate_intersection_point(p_H0_x_9, p_H1_x_9, x_values)
ent_grenze2 = calculate_intersection_point(p_H0_x_99, p_H1_x_99, x_values)

print(f"Die Entscheidungsgrenze bei P(H\u2080)=0.90 und P(H\u2081)=0.10 liegt bei: {round(ent_grenze1, 1)}")
print(f"Die Entscheidungsgrenze bei P(H\u2080)=0.99 und P(H\u2081)=0.01 liegt bei: {round(ent_grenze2, 1)}")

# Berechnung der Bayes Fehlerraten für Signifikanzniveaus 0.9 und 0.99
# Fehlerwahrscheinlichkeiten für Signifikanzniveau 0.9
false_positive_09, false_negative_09 = calculate_error_rates(ent_grenze1, mu_H0, std_H0, mu_H1, std_H1, p_H0[0], p_H1[0])

# Fehlerwahrscheinlichkeiten für Signifikanzniveau 0.01
true_negative_001, true_positive_001 = calculate_error_rates(ent_grenze2, mu_H0, std_H0, mu_H1, std_H1, p_H0[1], p_H1[1])

# Berechnung der Bayes Fehlerraten
bayes_error_rate_09 = false_positive_09 + false_negative_09
bayes_error_rate_001 = true_negative_001 + true_positive_001

print(f"Die Bayes-Fehlerrate für P(H\u2080)=0.90 und P(H\u2081)=0.10 liegt bei: {round(bayes_error_rate_09, 2)}")
print(f"Die Bayes-Fehlerrate für P(H\u2080)=0.99 und P(H\u2081)=0.01 liegt bei: {round(bayes_error_rate_001, 2)}")

# Wahrscheinlichkeitsfunktion und Posterioris (H0 = 0.9, H1 = 0.1)
plot_probability_functions(x_values, px_H0, px_H1, p_H0_x_9, p_H1_x_9, ent_grenze1,
                           title=r'Wahrscheinlichkeitsfunktion und Posterioris von $H_0=0.9$ und $H_1=0.1$',
                           filename='plots/Aufgabe3/b_1.png')

# Wahrscheinlichkeitsfunktion und Posterioris (H0 = 0.99, H1 = 0.01)
plot_probability_functions(x_values, px_H0, px_H1, p_H0_x_99, p_H1_x_99, ent_grenze2,
                           title=r'Wahrscheinlichkeitsfunktion und Posterioris von $H_0=0.99$ und $H_1=0.01$',
                           filename='plots/Aufgabe3/b_2.png')

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
# Berechnung alpha und 1-beta
False_P = 1 - stats.norm.cdf(x_values, loc=mu_H0, scale=std_H0)  # false positive
True_P = 1 - stats.norm.cdf(x_values, loc=mu_H1, scale=std_H1)  # true positive

# Plot der ROC Kurve
plt.figure(figsize=(18, 12))
plt.plot(False_P, True_P, color='green')
plt.title('ROC-Kurve (receiver operating characteristic)', fontsize=24)
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'$1 - \beta$', fontsize=20)
plt.grid(True)
plt.savefig('plots/Aufgabe3/c.png', format='png')
plt.show()

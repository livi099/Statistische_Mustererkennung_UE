# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

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

    Returns
    ----------
    df: Dichteverteilung (list)
    """
    temp1 = -(((x - mu) ** 2) / (2 * (sigma ** 2)))
    temp2 = (1 / (((2 * np.pi) ** (1 / 2)) * sigma)) * np.exp(temp1)
    return temp2

def randverteilung(c, p_x_omega1, p_x_omega2, p_omega1, p_omega2):
    """
    Berechnung der Radverteilung (evidence) des Merkmals

    Parameters
    ----------
    c: Klassen (list)
    p_x_omega1: Dichteverteilung (list)
    p_x_omega2: Dichteverteilung (list)
    p_omega1: prior (double)
    p_omega2: prior (double)

    Returns
    ----------
    evidence: Randverteilung (list)
    """
    evidence = []
    for i in range(len(c)):
        temp1 = p_x_omega1[i] * p_omega1 + p_x_omega2[i] * p_omega2
        evidence.append(temp1)
    return evidence

def posterior(c, p_omega, p_x_omega, p_x):
    """
    Berechnung des posteriors

    Parameters
    ----------
    c: Klassen (list)
    p_omega: prior (double)
    p_x_omega: Dichteverteilung (list)
    p_x: evidence (list)

    Returns
    ----------
    p_x : posterior (list)
    """
    p_pos = [(p_x_omega[i] * p_omega) / p_x[i] for i in range(len(c))]
    return p_pos

# Daten
x = np.arange(-20, 15, 0.01)

# Aufgabe 2a
P_OMEGA1, P_OMEGA2 = 0.3, 0.7  # a priori Wahrscheinlichkeit
MU1, MU2 = -7, 4
SIGMA1, SIGMA2 = np.sqrt(30), np.sqrt(13)

# Berechnung der Dichtefunktion für beide Klassen
p_x_omega1 = dichtefunktion(x, MU1, SIGMA1)
p_x_omega2 = dichtefunktion(x, MU2, SIGMA2)

# Berechnung der Randverteilung (evidence)
p_x = randverteilung(x, p_x_omega1, p_x_omega2, P_OMEGA1, P_OMEGA2)

# Berechnung des posterior
p_pos1 = posterior(x, P_OMEGA1, p_x_omega1, p_x)
p_pos2 = posterior(x, P_OMEGA2, p_x_omega2, p_x)

# Aufgabe 2b
# Bayes decicion rule
SAMPLE = [-15, -10, -5, 0, 5, 10]  # Stichprobe

sample_class = []
sample_posterior = []

for x_i in SAMPLE:
    min_dis = np.argmin(np.abs(x - x_i))
    pos1, pos2 = p_pos1[min_dis], p_pos2[min_dis]

    sample_class.append(1 if pos1 > pos2 else 2)
    sample_posterior.append(max(pos1, pos2))

# Visualisierung der Klassifizierung
table_data = []
for i in range(len(SAMPLE)):
    x_i = SAMPLE[i]
    assigned_class = sample_class[i]
    posterior_value = sample_posterior[i]

    if assigned_class == 1:
        class_description = '1'
    else:
        class_description = '2'

    table_data.append([f'X={x_i}', class_description, f'{posterior_value:.4f}'])

# Tabellarische Darstellung
headers = ["Merkmal", "Klasse", "Posterior"]
table = tabulate(table_data, headers, tablefmt="grid")

print(table)

# Aufgabe 2c
# Farbdefinitionen
color_omega1 = 'green'
color_omega2 = 'blue'
color_decision_boundary = 'black'
color_rand_distribution = 'red'

# Plot für Dichtefunktion
plt.figure(figsize=(10, 4))
plt.plot(x, p_x_omega1, label=r'$p(x|\omega_1)$', color=color_omega1)
plt.plot(x, p_x_omega2, label=r'$p(x|\omega_2)$', color=color_omega2)
plt.title('Dichtefunktion')
plt.grid(True)
plt.xlabel(r'$X$')
plt.xlim([-20, 15])
plt.ylabel(r'$p(x|\omega_i)$')
plt.legend()
plt.savefig('plots/Aufgabe2/a2_df.png', format='png')
plt.show()

# Finden des Schnittpunkts der beiden Kurven
intersection_index = np.argmin(np.abs(np.array(p_pos1) - np.array(p_pos2)))
intersection_point = x[intersection_index]

# Plot für Posteriors
plt.figure(figsize=(10, 4))
plt.plot(x, p_pos1, label=r'$p(\omega_1|x)$', color=color_omega1)
plt.plot(x, p_pos2, label=r'$p(\omega_2|x)$', color=color_omega2)
plt.axvline(x=intersection_point, color=color_decision_boundary, linestyle='--', label='Entscheidungsgrenze')
for i in range(len(SAMPLE)):
    x_i = SAMPLE[i]
    color = color_omega1 if sample_class[i] == 1 else color_omega2
    plt.scatter(x_i, sample_posterior[i], marker='o', color=color, s=60)
plt.title('Posteriors')
plt.grid(True)
plt.xlabel(r'$X$')
plt.xlim([-20, 15])
plt.ylabel(r'$p(\omega_i|x)$')
plt.legend()
plt.ylim([-0.1, 1.1])
plt.savefig('plots/Aufgabe2/a2_posterior.png', format='png')
plt.show()

# Plot für Randverteilung von X
plt.figure(figsize=(10, 4))
plt.plot(x, p_x, label=r'$p(x)$', color=color_rand_distribution)
plt.title('Randverteilung von X')
plt.grid(True)
plt.xlabel(r'$X$')
plt.xlim([-20, 15])
plt.ylabel(r'$p(x)$')
plt.legend()
plt.savefig('plots/Aufgabe2/a2_evidence.png', format='png')
plt.show()
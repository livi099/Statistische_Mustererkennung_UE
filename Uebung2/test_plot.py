import numpy as np
import matplotlib.pyplot as plt
import scipy

# Normalverteilung
mu = 4
var = 3
sigma = np.sqrt(var)

# Stichprobe
anzahl_SP = 1000  # Anzahl der Stichproben
umfang_SP = [5, 30, 100, 500]

# Generieren der Stichproben (normal-verteilt)
samples = dict()
for umfang in umfang_SP:
    data = np.zeros([anzahl_SP, umfang])
    for anzahl in range(anzahl_SP):
        data[anzahl, :] = np.random.normal(mu, sigma, umfang)
    samples[umfang] = data
    del data
del anzahl

SP_schätzer = dict()
SP_schätzer['mittel'], SP_schätzer['varianz'], SP_schätzer['median'] = dict(), dict(), dict()

for umfang in samples:
    mittel, varianz, median = np.zeros([anzahl_SP]), np.zeros([anzahl_SP]), np.zeros([anzahl_SP])
    for i, line in enumerate(samples[umfang]):
        mittel[i] = np.mean(line)
        varianz[i] = np.var(line)
        median[i] = np.median(line)
    SP_schätzer['mittel'][umfang], SP_schätzer['varianz'][umfang], SP_schätzer['median'][
        umfang] = mittel, varianz, median
    del mittel, varianz, median
del umfang, line, i

# Plot Histogramm der Mittelwerte für jeden Stichprobenumfang
plt.figure(figsize=(10, 6))  # Optional: Ändere die Größe des Plots
for umfang in SP_schätzer['mittel']:
    plt.hist(SP_schätzer['mittel'][umfang], bins=30, alpha=0.5, label=f'Umfang {umfang}')

    # Hinzufügen der Normalverteilung für jeden Datensatz
    mean = np.mean(samples[umfang])
    var = np.var(samples[umfang])
    x = np.linspace(min(SP_schätzer['mittel'][umfang]), max(SP_schätzer['mittel'][umfang]), 100)
    pdf = scipy.stats.norm.pdf(x, mu, var)
    plt.plot(x, pdf, linewidth=2, label=f'Normal Distribution (Umfang {umfang})')

plt.title('Histogramm der Mittelwerte für verschiedene Stichprobenumfänge mit Normalverteilung')
plt.xlabel('Mittelwert')
plt.ylabel('Häufigkeit')
plt.legend()
plt.show()

a = scipy.stats.norm.pdf(x, mu, var)
plt.plot(x, a)
plt.xlim([-10, 10])
plt.show()
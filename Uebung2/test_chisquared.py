import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Ursprüngliche Freiheitsgrade
df_original = 3

# Neue Freiheitsgrade für die Skalierung
df_scaled = 6

# Erzeugung von Daten für die x-Achse
x = np.linspace(0, 20, 1000)

# Berechnung der Wahrscheinlichkeitsdichte für beide Verteilungen
y_original = (chi2.pdf(x, df_original))
y_scaled = (chi2.pdf(x, df_original))/(df_original-1)

# Plot
plt.plot(x, y_original, label=f'Chi-Quadrat-Verteilung (df={df_original})')
plt.plot(x, y_scaled, label=f'Skalierte Chi-Quadrat-Verteilung (df={df_scaled})')
plt.title('Skalierte Chi-Quadrat-Verteilung')
plt.xlabel('X')
plt.ylabel('PDF')
plt.legend()
plt.show()
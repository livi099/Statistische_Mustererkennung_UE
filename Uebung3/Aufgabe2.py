# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def vek_rotieren(vektor, angle_deg):
    # Funktion rotiert einen Vektor um einen bestimmten Winkel
    # Winkel in Grad

    winkel_rad = np.radians(angle_deg)
    rot_matrix = np.array([[np.cos(winkel_rad), -np.sin(winkel_rad)],
                        [np.sin(winkel_rad), np.cos(winkel_rad)]])
    result = np.dot(rot_matrix, vektor)

    return result


def rayleigh_quotient(matrix, w_rot):
    # Funktion berechnet den Rayleigh-Quotienten
    rq = np.dot(np.dot(w_rot.T, matrix), w_rot) / np.dot(w_rot.T, w_rot)
    scale = rq * w_rot

    return rq, scale

# Kovarianz- und Korrelationsmatrix aus Aufgabe III.1
cov_matrix = [[23.4599939087314, -9.903475470150495], [-9.903475470150495, 7.9789076694751175]]
corr_matrix = [[1.0000000000000002, -0.7238556405746981], [-0.7238556405746981, 1.0]]

num_steps = 360
winkel = np.linspace(0, 360, num_steps)
rayleigh_cov = np.zeros(num_steps)
rayleigh_corr = np.zeros(num_steps)

# Initialisiere Arrays
scaled_directions_cov = np.zeros((num_steps, 2))
scaled_directions_corr = np.zeros((num_steps, 2))

# Richtungsvektor
initial_direction = np.array([1, 0])

for i, winkel in enumerate(winkel):
    rotated_direction = vek_rotieren(initial_direction, winkel)

    rayleigh_cov[i], scaled_directions_cov[i, :] = rayleigh_quotient(cov_matrix, rotated_direction)
    rayleigh_corr[i], scaled_directions_corr[i, :] = rayleigh_quotient(corr_matrix, rotated_direction)


# Eigenwertzerlegung
eigenwerte_cov, eigenvektoren_cov = linalg.eig(cov_matrix)
print(eigenvektoren_cov)
eigenvektoren_cov = eigenvektoren_cov.T

eigenwerte_corr, eigenvektoren_corr = linalg.eig(corr_matrix)
eigenvektoren_corr = eigenvektoren_corr.T


# Plot
plt.figure(figsize=(10, 6))
plt.plot(scaled_directions_cov[:, 0], scaled_directions_cov[:, 1],
           color='orange', label='Skalierte Richtungsvektoren')
for i in range(len(eigenwerte_cov)):
    scaled_eigenvector = eigenvektoren_cov[i] * np.sqrt(eigenwerte_cov[i])
    print(scaled_eigenvector)
    color = ['blue', 'green']
    plt.plot([0, scaled_eigenvector[0]], [0, scaled_eigenvector[1]], linestyle='--', color=color[i],
             label=f'Skalierter Eigenvektor {i + 1}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('plots/Aufgabe2/cov.eps', format='eps')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(scaled_directions_corr[:, 0], scaled_directions_corr[:, 1],
            color='orange', label='Skalierte Richtungsvektoren')
for i in range(len(eigenwerte_corr)):
    scaled_eigenvector = eigenvektoren_corr[i] * np.sqrt(eigenwerte_corr[i])
    color = ['blue', 'green']
    plt.plot([0, scaled_eigenvector[0]], [0, scaled_eigenvector[1]], linestyle='--', color=color[i],
             label=f'Skalierter Eigenvektor {i + 1}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('plots/Aufgabe2/corr.eps', format='eps')
plt.show()
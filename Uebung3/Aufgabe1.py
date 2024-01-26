# Statistische Musterekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy.linalg as linalg
import numpy as np

# Aufgabe a)
np.random.seed(98)

# Kovarianzmatrix
mu = [4, 7]
var = [12, 2]
rho = -0.5
cov = [[var[0], rho * np.sqrt(var[0]*var[1])], [rho * np.sqrt(var[1]*var[0]), var[1]]]

# Eigenwertzerlegung
eigenwerte, eigenvektoren = linalg.eig(cov)

# (Inverse) whitening-Transformation
lam = np.diag(eigenwerte**-0.5)
wt = np.dot(lam, eigenvektoren.T)
wt_inv = np.linalg.inv(wt)

# Stichprobe
stichp = np.random.normal(0, 1, 600).reshape((2, 300))

# Inverse whitening-Transformation wird auf Stichprobe angewendet
stichp_wt = np.dot(wt_inv, stichp)

#------------------------------------------------------------------------
# Aufgabe b)
stichp_wt[0] = stichp_wt[0]+mu[0]
stichp_wt[1] = stichp_wt[1]+mu[1]

# Schätzung des Mittelwerts
mu_stichp_wt1 = sum(stichp_wt[0]) / len(stichp_wt[0])
mu_stichp_wt2 = sum(stichp_wt[1]) / len(stichp_wt[1])


var1 = sum((stichp_wt[0]-mu_stichp_wt1)*((stichp_wt[0]-mu_stichp_wt1).T))/(len(stichp_wt[0])-1)
var2 = sum((stichp_wt[1]-mu_stichp_wt2)*((stichp_wt[1]-mu_stichp_wt2).T))/(len(stichp_wt[1])-1)
cov = sum((stichp_wt[0]-mu_stichp_wt1)*((stichp_wt[1]-mu_stichp_wt2).T))/(len(stichp_wt[0])-1)

cov_matrix = [[var1, cov], [cov, var2]]
corr_matrix = [[var1/np.sqrt(var1*var1), cov/np.sqrt(var1*var2)], [cov/np.sqrt(var2*var1), var2/np.sqrt(var2*var2)]]

#------------------------------------------------------------------------
# Aufgabe c)
# Berechnung mit Numpy-Package für einen Vergleich
cov_matrix_pack = np.cov(stichp_wt)
corr_matrix_pack = np.corrcoef(stichp_wt)


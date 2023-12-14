# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate


# Aufgabe3

def dichte(x, mu, std):
    p_x=[]
    for i in x:
        p_x1 = 1/(np.sqrt(2*np.pi)*std) * np.exp(-np.power(i-mu,2)/(2*np.power(std,2))) # S. 237
        p_x.append(p_x1)
    return p_x

def rand(x, p_x_mu, p_mu, p_x_mu2, p_mu2 ):
    p_x=[]
    for i in range(len(x)):
        p_x1=p_x_mu[i] * p_mu +p_x_mu2[i] * p_mu2
        p_x.append(p_x1)
    return p_x

    return

def posterior(x, p_x_mu, p_mu, p_x):
    p_mu_x=[]
    for i in range(len(x)):
        p_mu_x1=(p_mu[i] * p_x_mu) / p_x[i]
        p_mu_x.append(p_mu_x1)
    return p_mu_x

# Werte
p_H0 = [0.9, 0.99]
p_H1 = [0.1, 0.01]
mu_H0 = 4
std_H0 = np.sqrt(1)
mu_H1 = 5
std_H1 = np.sqrt(1)

# Aufgabe a)
x_values=np.linspace(0, 12, 1000)

False_P = 1-stats.norm.cdf(x_values, loc=mu_H0, scale=std_H0)  # false positive
True_P = 1-stats.norm.cdf(x_values, loc=mu_H1, scale=std_H1)  # true positive

VW1=[]
VW2=[]
for i, j in enumerate(x_values):
    a = (True_P[i] * p_H1[0]) / (True_P[i] * p_H1[0] + False_P[i] * p_H0[0])
    b = (True_P[i] * p_H1[1]) / (True_P[i] * p_H1[1] + False_P[i] * p_H0[1])
    VW1.append(a)
    VW2.append(b)


plt.figure(figsize=(18, 12))
plt.plot(x_values, VW1, color='orange', label=r'$H_0$ = 0.9, $H_1$ = 0.1')
plt.plot(x_values, VW2, color='brown', label=r'$H_0$ = 0.99, $H_1$ = 0.01')
#plt.title(r'positiven Vorhersagewert als Funktion der Entscheidungsgrenze')
plt.ylabel(r'$p(H_1|+)$')
plt.xlabel(r'$x^*$')
plt.legend()
plt.grid(True)
plt.savefig('plots/Aufgabe3/a.eps', format='eps')
plt.show()


# Aufgabe b)
px_H0 = dichte(x_values, mu_H0, std_H0)
px_H1 = dichte(x_values, mu_H1, std_H1)
px_9 = rand(x_values, px_H0, p_H0[0], px_H1, p_H1[0])
px_99 = rand(x_values, px_H0, p_H0[1], px_H1, p_H1[1])
p_H0_x_9 = posterior(x_values,p_H0[0], px_H0, px_9)
p_H1_x_9 = posterior(x_values,p_H1[0], px_H1, px_9)
p_H0_x_99 = posterior(x_values,p_H0[1], px_H0, px_99)
p_H1_x_99 = posterior(x_values,p_H1[1], px_H1, px_99)

# Schnittmenge finden
round1 = [round(zahl, 5) for zahl in p_H0_x_9]
round2 = [round(zahl, 5) for zahl in p_H1_x_9]
gemeinsame_elemente = set(round1).intersection(round2)
gemeinsame_elemente = [x for x in round1 if x in round2]
print(list(gemeinsame_elemente))
P1_err_H0 = 1-stats.norm.cdf(6.7, loc=mu_H0, scale=std_H0)*p_H0[0]
P2_err_H0 = stats.norm.cdf(6.7, loc=mu_H1, scale=std_H1)*p_H1[0]
R1 = 1-stats.norm.cdf(9.1, loc=mu_H0, scale=std_H0)*p_H0[1]
R2 = stats.norm.cdf(9.1, loc=mu_H1, scale=std_H1)*p_H1[1]

BFR_9 = P1_err_H0 + P2_err_H0
BFR_99 = R1 + R2

print(BFR_9)
print(BFR_99)

plt.figure(figsize=(18, 12))
plt.plot(x_values, p_H0_x_9, label=r'$H_0$ = 0.9, $H_1$ = 0.1', color='orange')
plt.plot(x_values, p_H1_x_9, color='orange')
plt.plot(x_values, p_H0_x_99, color='brown')
plt.plot(x_values, p_H1_x_99, label=r'$H_0$ = 0.99, $H_1$ = 0.01', color='brown')
plt.plot([6.7, 6.7], [0, 1], color='k')
plt.plot([9.1, 9.1], [0, 1], label='Entscheidungsgrenzen', color='k')
plt.title('Posteriors')
plt.grid(True)
plt.ylabel(r'$p(H_i|x)$')
plt.xlabel(r'$x$')
plt.legend()
idx1 = np.argwhere(np.diff(np.sign(np.array(p_H0_x_9) - np.array(p_H1_x_9)))).flatten()
idx2 = np.argwhere(np.diff(np.sign(np.array(p_H0_x_99) - np.array(p_H1_x_99)))).flatten()
print(idx1)
print(idx2)
print(x_values[514])
print(x_values[698])
plt.savefig('plots/Aufgabe3/b_3.eps', format='eps')
plt.show()



# Aufgabe c)
plt.figure(figsize=(18, 12))
plt.plot(False_P, True_P, color='green')
#plt.title(r'ROC-Kurve')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$1 - \beta$')
plt.grid(True)
plt.savefig('plots/Aufgabe3/c.eps', format='eps')
plt.show()



#Test
plt.figure(figsize=(18, 12))
p_x_H0 = dichte(x_values, mu_H0, std_H0)
p_x_H1 = dichte(x_values, mu_H1, std_H1)
plt.plot(x_values, p_x_H0, label=r'$H_0$ - gesund', color='red', linestyle='--')
plt.plot(x_values, p_x_H1, label=r'$H_1$ - infiziert', color='blue', linestyle='--')
plt.plot(x_values, p_H0_x_9, label=r'$H_0$ = 0.9', color='red')
plt.plot(x_values, p_H1_x_9,label=r'$H_1$ = 0.1', color='blue')
plt.plot([6.7, 6.7], [0, 1], label='Entscheidungsgrenzen', color='k')
plt.title(r'Wahrscheinlichkeitsfuntion $H_0$ und $H_1$')
plt.grid(True)
plt.ylabel(r'$p(x|H_i)$')
plt.xlabel(r'$x$')
plt.legend()
plt.savefig('plots/Aufgabe3/b_1.eps', format='eps')
plt.show()

#Test
plt.figure(figsize=(18, 12))
p_x_H0 = dichte(x_values, mu_H0, std_H0)
p_x_H1 = dichte(x_values, mu_H1, std_H1)
plt.plot(x_values, p_x_H0, label=r'$H_0$ - gesund',color='red', linestyle='--')
plt.plot(x_values, p_x_H1, label=r'$H_1$ - infiziert', color='blue', linestyle='--')
plt.plot(x_values, p_H0_x_99, label=r'$H_0$ = 0.99', color='red')
plt.plot(x_values, p_H1_x_99,label=r'$H_1$ = 0.01', color='blue')
plt.plot([9.1, 9.1], [0, 1], label='Entscheidungsgrenzen', color='k')
plt.title(r'Wahrscheinlichkeitsfuntion $H_0$ und $H_1$')
plt.grid(True)
plt.ylabel(r'$p(x|H_i)$')
plt.xlabel(r'$x$')
plt.legend()
plt.savefig('plots/Aufgabe3/b_2.eps', format='eps')
plt.show()
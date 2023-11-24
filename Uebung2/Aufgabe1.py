# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import random
from scipy.special import gamma
from scipy.stats import chi2
import seaborn as sns


# Aufgabe 1

def create_subplots(sample_size, bootstrap_samples, mu, std):
    plt.figure(figsize=(18, 12))
    x_values = np.linspace(0, 10, 10000)
    for i, estimator in enumerate( ['Stichprobenmittel', 'Stichprobenvarianz', 'Stichprobenmeridian']):
        statistics = calculate_statistics(bootstrap_samples[i],0)
        j=0
        plt.gca()
        for s in statistics:

            ax=plt.subplot(3, 1, j + 1)

            ax.hist(s, bins=100, alpha=0.6, density=False, label='Bootstrap aus Stichprobe: ' + str(i))

            #plt.set_title(f'{stat} - Stichprobengröße {sample_size}')
            #plt.xlim([3.5,4.5])

            if estimator == 'Stichprobenmittel' and j==0:

                ax2 = ax.twinx()

                x_values2 = (x_values - mu) / (var / size)
                p_x = 1 / np.sqrt(2 * np.pi) * np.exp(-x_values2 ** 2 / 2) / (var / size)
                p_x=norm.pdf(x_values, loc=mu)

                #tmp = -(np.power(x_values - mu, 2)) / (2 * np.power(std, 2))
                #p_x = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(tmp)

                ax2.plot(x_values, p_x, color=colorplots[j], linestyle='--')
                ax2.set_ylabel('p(x)')
                ax2.set_ylim([0, 6])
            j=j+1
            ax.set_title(f'{estimator}')
            ax.set_xlabel('Schätzwert')
            ax.set_ylabel('Häufigkeit')
            #ax.set_xlim([2, 6])
            ax.set_ylim([0, 60])
            ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def calculate_statistics(samples,t):
    means = np.sum(samples, axis=1) / np.size(samples, axis=1)
    var_sums = (samples - means[:, np.newaxis]) ** 2
    if t == 0:
        vars = np.sum(var_sums, axis=1) / np.size(var_sums, axis=1)
    elif t == 1:
        vars = np.sum(var_sums, axis=1) / np.size(var_sums-1, axis=1)
    else:
        print('Nur 0 oder 1 möglich!')

    medians = np.median(samples, axis=1)
    return means, vars, medians

# Seed für die Reproduzierbarkeit der Zufallszahlen setzen
np.random.seed(235)

# Stichproben
num_samples = 1000
sample_sizes = [5, 30, 100, 500]

# Normalverteiung
mu = 4
var = 3
std = np.sqrt(var)


# Stichproben generieren
stichp = {}

# Aufgabe a)

# DataFrame für die Schätzer erstellen
df_estimators = pd.DataFrame()

# Schleife über die Größen
for size in sample_sizes:
    # Stichproben generieren
    samples = np.random.normal(loc=mu, scale=std, size=(num_samples, size))

    # Stichprobemittel, Stichprobenvarianz und Stichprobenmedian berechnen
    sample_means,sample_variances,sample_medians= statistics = calculate_statistics(samples,0)

    # DataFrame für die aktuellen Schätzer erstellen
    df_size = pd.DataFrame({
        'Size': size,
        'Stichprobenmittel': sample_means,
        'Stichprobenvarianz': sample_variances,
        'Stichprobenmeridian': sample_medians
    })

    # Der DataFrame df_estimators die neuen Daten hinzufügen
    df_estimators = pd.concat([df_estimators, df_size])

colorplots = ['green', 'blue', 'red', 'orange']

plt.figure(figsize=(18, 12))

x_values = np.linspace(0, 10, 10000) #????????


# Schleife über die Schätzer

for i, estimator in enumerate(['Stichprobenmittel', 'Stichprobenvarianz', 'Stichprobenmeridian'], start=1):
    j = 0
    ax = plt.subplot(3, 1, i)
    for size in sample_sizes:

        # plt.hist(df_estimators[df_estimators['Size'] == size][estimator], bins=100, alpha=0.5, color=colorplots[j],
        #          edgecolor=colorplots[j], label='Stichprobengröße: ' + str(size))
        #plt.grid(True)
        #plt.set_axisbelow(True)
        ax.hist(df_estimators[df_estimators['Size'] == size][estimator], bins=100, alpha=0.4, label='Umfang der Stichprobe: ' + str(size),
                color=colorplots[j])
        plt.suptitle('Empirische Verteilung dreier Schätzer mit wahrer Dichteverteilung')
        ax.grid(True)
        ax.set_axisbelow(True)
        # Wahre Dichtefunktion plotten
        if estimator == 'Stichprobenmittel':
            ax2 = ax.twinx()

            x_values2=(x_values-mu)/(var /size)
            p_x=1/np.sqrt(2*np.pi) * np.exp(-x_values2**2/2)/(var / size)
            #p_x=norm.pdf(x_values, loc=mu, scale=var / size)

            ax2.plot(x_values, p_x, color=colorplots[j], linestyle='--')
            ax2.set_ylabel('p(x)')
            ax2.set_ylim([0, 6])

        elif estimator == 'Stichprobenvarianz':
            ax2 = ax.twinx()

            n = (size - 1)
            #p_x = chi2.pdf(x_values,size-1,scale=var/n)

            x_values2 = (x_values) / (var / n)
            a = 1 / (np.power(2, (n / 2)) * gamma(n / 2))
            p_x=(a*np.power(x_values2, (n / 2 - 1))*np.exp(-x_values2 / 2)) / (var/n)

            ax2.plot(x_values, p_x, color=colorplots[j], linestyle='--')
            ax2.set_ylabel('p(x)')
            ax2.set_ylim([0, 1])
        j = j + 1
        ax.set_title(f'{estimator}')
        ax.set_xlabel('Schätzwert')
        ax.set_ylabel('Häufigkeit')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 70])
        ax.legend()

    # --------------------------------------------------------------------------------------------------------------------
    # Aufgabe b)
    for size in sample_sizes:
        values = df_estimators[df_estimators['Size'] == size][estimator]
        mean = np.sum(values) / len(values)
        std = np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1))

        print(f"{estimator} - Size {size} - Mittelwert aus Realisierungen: {mean:.4f}, Standardabweichung aus Realisierungen: {std:.4f}")

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------
# Aufgabe c)

# Z-Wert für 95%-Konfidenzintervall (zweiseitig)
z_value = stats.norm.ppf(0.975)
population_variance = 3
# Grenze für das wahre Mittel
true_mean_limit = 4

for size in sample_sizes:
    # Standardabweichung der Stichprobenmittel berechnen
    standard_error = np.sqrt(population_variance / size)

    # Konfidenzintervall berechnen
    confidence_interval_lower = df_estimators[df_estimators['Size'] == size]['Stichprobenmittel'] - z_value * standard_error
    confidence_interval_upper = df_estimators[df_estimators['Size'] == size]['Stichprobenmittel'] + z_value * standard_error

    # Zählen, wie viele Konfidenzintervalle das wahre Mittel 4 übersteigen
    exceeding_intervals = np.sum((confidence_interval_lower < 4) & (confidence_interval_upper > 4))

    # Anteil berechnen und ausgeben
    exceeding_ratio = (exceeding_intervals / num_samples) * 100
    print(f"Stichprobengröße {size}: Anteil überschrittener Konfidenzintervalle: {exceeding_ratio:.4f}")


for size in sample_sizes:
    t_value = stats.t.ppf(0.975, df=size-1)

    # Standardabweichung der Stichprobenmittel berechnen
    standard_error = np.sqrt(df_estimators[df_estimators['Size'] == size]['Stichprobenvarianz'] / (size-1)) #???????????????? size-1?

    # Konfidenzintervall berechnen
    confidence_interval_lower = df_estimators[df_estimators['Size'] == size]['Stichprobenmittel'] - z_value * standard_error
    confidence_interval_upper = df_estimators[df_estimators['Size'] == size]['Stichprobenmittel'] + z_value * standard_error

    # Zählen, wie viele Konfidenzintervalle das wahre Mittel 4 übersteigen
    exceeding_intervals = np.sum((confidence_interval_lower < 4) & (confidence_interval_upper > 4))

    # Anteil berechnen und ausgeben
    exceeding_ratio = (exceeding_intervals / num_samples) * 100
    print(f"Stichprobengröße {size}: Anteil überschrittener Konfidenzintervalle: {exceeding_ratio:.4f}")


#-------------------------------------------------------------------------------------------------------------------

# Aufgabe d)

# Erzeuge 3 verschiedene Stichproben für jeden Umfang mit je 1000 Bootstrap-Stichproben
#bootstrap_samples = {size: [[np.random.normal(loc=mu, scale=std, size=size) for _ in range(3)] for _ in range(1000)]  for size in sample_sizes}

bootstrap_samples = dict()
for size in sample_sizes:
    bootstrap_samples[size]=dict()
    for num_stichprobe in range(3):
        stdprob = (np.random.normal(mu, std, size))
        data_boot = np.zeros([num_samples, size])
        for anzahl in range(num_samples):
            data_boot[anzahl, :] = random.choices(stdprob, k=size)
        bootstrap_samples[size][num_stichprobe] = data_boot


# Erstelle Subplots für jede Stichprobengröße
for size in sample_sizes:
    create_subplots(size, bootstrap_samples[size], mu, std)
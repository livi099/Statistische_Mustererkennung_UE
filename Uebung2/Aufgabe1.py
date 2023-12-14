# Statistische Muserekennung WS 2023
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

# import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import random
from scipy.special import gamma
import warnings
from scipy.stats import chi2



# Aufgabe 1

# suppress warnings
warnings.filterwarnings('ignore')

def plot_rand(name, data, sizes_samp, colors, var):
    plt.figure(figsize=(18, 12))
    x_values = np.linspace(0, 10, 10000)
    for i, estimator in enumerate(['Stichprobenmittel', 'Stichprobenvarianz', 'Stichprobenmeridian'], start=1):
        j = 0
        ax = plt.subplot(3, 1, i)
        for size in sizes_samp:

            ax.hist(data[data['Size'] == size][estimator], bins=100, alpha=0.5,
                    label='Stichprobengröße: ' + str(size), color=colors[j])
            # plt.suptitle('Empirische Verteilung dreier Schätzer mit wahrer Dichteverteilung')
            ax.grid(True)
            ax.set_axisbelow(True)

            # Wahre Dichtefunktion
            if estimator == 'Stichprobenmittel':
                ax2 = ax.twinx()

                x_values2 = (x_values - mu) / (var / size)
                p_x = 1 / np.sqrt(2 * np.pi) * np.exp(-x_values2 ** 2 / 2) / (var / size)
                # p_x=norm.pdf(x_values, loc=mu, scale=var / size)

                ax2.plot(x_values, p_x, color=colors[j], linestyle='--')
                ax2.set_ylabel('p(x)')
                ax2.set_ylim([0, 6])

            elif estimator == 'Stichprobenvarianz':
                ax2 = ax.twinx()

                n = (size - 1)
                # p_x = chi2.pdf(x_values,size-1,scale=var/n)

                x_values2 = (x_values) / (var / n)
                a = 1 / (np.power(2, (n / 2)) * gamma(n / 2))
                p_x = (a * np.power(x_values2, (n / 2 - 1)) * np.exp(-x_values2 / 2)) / (var / n)

                ax2.plot(x_values, p_x, color=colors[j], linestyle='--')
                ax2.set_ylabel('p(x)')
                ax2.set_ylim([0, 1])
            j = j + 1
            ax.set_title(f'{estimator}')
            if estimator == "Stichprobenmeridian":
                ax.set_xlabel('Schätzwert')
            ax.set_ylabel('Häufigkeit')
            ax.set_xlim([0, 10])
            ax.set_ylim([0, 70])
            ax.legend()
            plt.savefig('plots/Aufgabe1/emp_ver_' + name +'.png', format='png')

        # --------------------------------------------------------------------------------------------------------------------
        # Aufgabe b)
        for size in sizes_samp:
            values = data[data['Size'] == size][estimator]
            mean = np.sum(values) / len(values)
            std = np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1))

            print(
                f"{estimator} - Size {size} - Mittelwert aus Realisierungen: {mean:.4f}, Standardabweichung aus Realisierungen: {std:.4f}")

    plt.tight_layout()
    plt.show()

def plot_boot(size, df, mu, var, number_str):
    plt.figure(figsize=(18, 12))
    x_values = np.linspace(0, 10, 1000)

    for i, estimator in enumerate(['Stichprobenmittel', 'Stichprobenvarianz', 'Stichprobenmeridian'], start=1):

        j = 0
        ax = plt.subplot(3, 1, i)
        for i in range(number_str):

            ax.hist(df[df['Stichp'] == i][estimator], bins=100, alpha=0.5, density=False, label='Bootstrap aus Stichprobe: ' + str(i), color=colors[j])
            j = j + 1

        # Wahre Dichtefunktion
        if estimator == 'Stichprobenmittel':
            ax2 = ax.twinx()

            x_values2 = (x_values - mu) / (var / size)
            p_x = 1 / np.sqrt(2 * np.pi) * np.exp(-x_values2 ** 2 / 2) / (var / size)
            # p_x=norm.pdf(x_values, loc=mu, scale=var / size)

            ax2.plot(x_values, p_x, color=colors[j], linestyle='--')
            ax2.set_ylabel('p(x)')
            ax2.set_ylim([0, 6])

        elif estimator == 'Stichprobenvarianz':
            ax2 = ax.twinx()

            n = (size - 1)
            # p_x = chi2.pdf(x_values,size-1,scale=var/n)

            x_values2 = (x_values) / (var / n)
            a = 1 / (np.power(2, (n / 2)) * gamma(n / 2))
            p_x = (a * np.power(x_values2, (n / 2 - 1)) * np.exp(-x_values2 / 2)) / (var / n)

            ax2.plot(x_values, p_x, color=colors[j], linestyle='--')
            ax2.set_ylabel('p(x)')
            ax2.set_ylim([0, 1])

        ax.set_title(f'{estimator}')
        if estimator == "Stichprobenmeridian":
            ax.set_xlabel('Schätzwert')
        ax.set_ylabel('Häufigkeit')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 70])
        ax.legend()
        plt.savefig('plots/Aufgabe1/bootstrap_' + str(size) +'.png', format='png')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def calculate_statistics(samples,t):
    means = np.sum(samples, axis=1) / np.size(samples, axis=1)
    var_sums = (samples - means[:, np.newaxis]) ** 2
    if t == 0:
        vars = np.sum(var_sums, axis=1) / np.size(var_sums, axis=1) # wenn wahres Mittel bekannt
    elif t == 1:
        vars = np.sum(var_sums, axis=1) / (np.size(var_sums, axis=1)-1) ## wenn wahres Mittel nicht bekannt
    else:
        print('Nur 0 oder 1 möglich!')

    medians = np.median(samples, axis=1)
    return means, vars, medians

# Seed für die Reproduzierbarkeit der Zufallszahlen setzen


np.random.seed(234) 


# Stichproben
num_samples = 1000
sizes_samp = [5, 30, 100, 500]

# Normalverteiung
mu = 4
var = 3
std = np.sqrt(var)


# Aufgabe a) und b)

# DataFrame für die Schätzer erstellen
df_rand = pd.DataFrame()

# Schleife über die Größen
for size in sizes_samp:

    # Stichproben generieren
    samples = np.random.normal(loc=mu, scale=std, size=(num_samples, size))

    # Stichprobemittel, Stichprobenvarianz und Stichprobenmedian berechnen
    sample_means,sample_variances,sample_medians= calculate_statistics(samples,1)

    # DataFrame erstellen
    df_size = pd.DataFrame({
        'Size': size,
        'Stichprobenmittel': sample_means,
        'Stichprobenvarianz': sample_variances,
        'Stichprobenmeridian': sample_medians
    })

    # Neue Daten zum Dataframe hinzufügen
    df_rand = pd.concat([df_rand, df_size])

colors = ['green', 'blue', 'red', 'orange']

name = 'rand'
plot_rand(name, df_rand, sizes_samp, colors, var)

#--------------------------------------------------------------------------------------------------------------------
# Aufgabe c)


pop_variance = 3
limit = 4 # Grenze für das wahre Mittel
interval = 0.95

for size in sizes_samp:
    z_value = stats.norm.ppf((1 + 0.95) / 2)
    
    # Standard Fehler 
    standard_error = np.sqrt(pop_variance / size)

    # Konfidenzintervall (untere und obere Grenze)
    lower = df_rand[df_rand['Size'] == size]['Stichprobenmittel'] - z_value * standard_error
    upper = df_rand[df_rand['Size'] == size]['Stichprobenmittel'] + z_value * standard_error

    # Zählen, wie viele Konfidenzintervalle das wahre Mittel 4 übersteigen
    exc_intervals = np.sum((lower < 4) & (upper > 4))

    exceeding_ratio = (exc_intervals / num_samples) * 100 # in %
    print(f"Stichprobengröße {size}: Anteil überschrittener Konfidenzintervalle: {exceeding_ratio:.4f}")


for size in sizes_samp:
    t_value = stats.t.ppf((1 + interval) / 2, df=size-1)

    # Standard Fehler 
    standard_error = np.sqrt(df_rand[df_rand['Size'] == size]['Stichprobenvarianz'] / (size-1)) #???????????????? size-1?

    # Konfidenzintervall (untere und obere Grenze)
    lower = df_rand[df_rand['Size'] == size]['Stichprobenmittel'] - z_value * standard_error
    upper = df_rand[df_rand['Size'] == size]['Stichprobenmittel'] + z_value * standard_error

    # Zählen, wie viele Konfidenzintervalle das wahre Mittel 4 übersteigen
    exc_intervals = np.sum((lower < 4) & (upper > 4))

    exceeding_ratio = (exc_intervals / num_samples) * 100 # in %
    print(f"Stichprobengröße {size}: Anteil überschrittener Konfidenzintervalle: {exceeding_ratio:.4f}")


#-------------------------------------------------------------------------------------------------------------------

# Aufgabe d)

# Generie bootstrap samples aus 1 und 3 Stichproben
number_str=[1, 3]
df_boot = pd.DataFrame()

for k in number_str:
    for size in sizes_samp:
        for num_stichprobe in range(k):
            a=[]
            stprob = (np.random.normal(mu, std, size))
            for anzahl in range(num_samples):
                test= random.choices(stprob, k=size)
                a.append(test)

            # Stichprobemittel, Stichprobenvarianz und Stichprobenmedian berechnen
            sample_means, sample_variances, sample_medians = calculate_statistics(a, 1)

            # DataFrame für die aktuellen Schätzer erstellen
            df_size = pd.DataFrame({
                'Stichp': int(num_stichprobe),
                'Size': size,
                'Stichprobenmittel': sample_means,
                'Stichprobenvarianz': sample_variances,
                'Stichprobenmeridian': sample_medians
            })

            # Neue Daten zum Dataframe hinzufügen
            df_boot = pd.concat([df_boot, df_size])

    # Plot für 1 Stichprobe
    if k==1:
        name='boot'
        plot_rand(name, df_boot, sizes_samp, colors, var)

    # Plot für 3 Stichproben
    if k==3:
        for size in sizes_samp:
            plot_boot(size, df_boot.loc[df_boot['Size']==size], mu, var, k)
# Statistische Muserekennung WS 2023
# Laurin Koppenwallner, 11726954
# Benjamin Stifter, 01618881
# Olivia Panzenböck, 11775488

import numpy as np
import pandas as pd

# Aufgabe 2
# a

# Wahrscheinlichkeit beim Wurf eines sechseitigen Würfels
p = 1/6

# Vereinigungsmenge A U B gibt an, dass entweder A oder B eintritt

p_A = 3 * p # 4, 5, 6
p_B = 3 * p # 2, 4, 6
p_AB_durchschnitt = 2 * p # 4 ,6

# Summenregel
p_AB_vereinigt = p_A + p_B - p_AB_durchschnitt

print(p_AB_vereinigt)

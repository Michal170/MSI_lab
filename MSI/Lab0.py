# Zaimportuj bibliotekę numpy

import numpy as np

# Utwórz tablicę numpy w zmiennej o wymiarach 3x5 wypełnioną jedynkami. Przypiszą ją do zmiennej `A`.

A = np.ones([3,5])
print(f"Tablica:\n {A}")
# Zsumuj tablicę `A` po rzędach i kolumnach. Wyświetl wynik.

column_sum = np.sum(A,axis=0)
rows_sum = np.sum(A,axis=1)
print(f"Rows:{rows_sum}")
print(f"Columns:{column_sum}")

# Utwórz jednowymiarową tablicę numpy o 20 elementach wypełnioną kolejnymi liczbami naturalnymi zaczynając od 0. Przypisz ją do zmiennej `B`

B = np.arange(20)

# Wyświetl elementy tablicy `B` od dziesiątego do dwunastego włącznie.

print(f"B[9:12]:{B[9:12]}")

# Zaimportuj biblioteki pandas oraz sklearn

import pandas as pd
import sklearn as sk
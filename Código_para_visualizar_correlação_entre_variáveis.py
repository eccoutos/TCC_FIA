# Importando Modulos e Pacotes
import numpy as np
import pandas as pd

# Atribuindo parametros de entrada
target = 'result'

# Importando os dados de entrada
df = pd.read_csv('C:/Users/rcosin/Desktop/FIA/TCC/Rim/chronic_kidney_disease_full.csv', sep = ',', decimal = '.')

import matplotlib.pyplot as plt

df.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
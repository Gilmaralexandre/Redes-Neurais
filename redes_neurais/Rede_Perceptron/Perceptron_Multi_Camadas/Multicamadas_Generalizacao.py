#!/usr/bin/python
# -*- coding: utf-8 -*-

# Esse script serve para testar a generalização da rede neural treinada no
# script Multicamadas.py
#
# OBS: É necessário executar o Multicamadas.py ANTES de executar esse aqui!
#
# Author: João Marcos Meirelles da Silva
# creation date	: jan, 23th, 2019
# updated	: aug, 28th, 2019

#import numpy as np

# Carrega os dados de treinamento
from matplotlib import colors

peso = np.array([110, 113, 120,  125, 97])
pH   = np.array([6.0, 4.4, 3.5, 5.5, 5.0])

# Vetor de classificação desejada.
d = np.array([-1, 1, 1, -1, 1])

# Normalização das entradas
pesoN = peso / peso.max()
pHN   = pH / pH.max()

# Deslocamento para a origem
pesoN = pesoN - pesoN.mean()
pHN   = pHN - pHN.mean()

# Entrada do Perceptron.
X = np.vstack((pesoN, pHN))   # Ou X = np.asarray([pesoN, pHN])

# ===============================================================
# TESTE DA REDE.
# ===============================================================

Error_Test = np.zeros(5)

for i in range(5):
    # Insere o bias no vetor de entrada.
    Xb = np.hstack((bias, X[:,i]))

    # Saída da Camada Escondida.
    O1 = np.tanh(W1.dot(Xb))            # Equações (1) e (2) juntas.      

    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    O1b = np.insert(O1, 0, bias)

    # Neural network output
    Y = np.tanh(W2.dot(O1b))            # Equações (3) e (4) juntas.

    Error_Test[i] = d[i] - np.round(Y)
    
print(Error_Test)

# Plota os dados originais - Debugging,
plt.scatter(peso, pH, c=d, marker='^', cmap=colors.ListedColormap(['red', 'orange']))
plt.xlabel('peso')
plt.ylabel('pH')

# Carrega os dados de treinamento
peso = np.array([113, 122, 107,  98, 115, 120, 104, 108, 117, 101, 112, 106, 116])
pH   = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2, 6.3, 4.0, 6.3, 4.2, 5.6, 3.1, 5.0])
d = np.array([-1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1])

plt.scatter(peso, pH, c=d, cmap=colors.ListedColormap(['red', 'orange']))
plt.show()

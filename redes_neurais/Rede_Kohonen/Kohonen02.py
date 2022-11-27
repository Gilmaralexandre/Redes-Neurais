# -*- coding: utf-8 -*-
"""
Esse script utiliza a Regra Delta para treinar a rede de kohonen. Os
dados de entrada para treinamento são os da base de dados iris.data.
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

# Taxa de aprendizado. 0 < alfa < 1.
alfa = 0.001

# Número de épocas
numEpocas = 50

# Numerto de classes.
K = 3

# Raio das classes.
r = 2

# Número de elementos dados da base.
N = 150

# Metade da base será utilizada para treinamento.
Meio_N = 75

# Ordena os índices dos dados aleatoriamente.
ordem_elementos = np.random.permutation(N)

# Conjunto de dados de treinamento (pelo índice).
Treinamento = ordem_elementos[:Meio_N]

# Conjunto de dados de teste (pelo índice)
Teste = ordem_elementos[Meio_N:]

# Inicia o vetor que armazena a classificação de cada dado da base iris2.
Classificacao = np.zeros(Meio_N)

Classificacao_Original = np.copy(Classificacao)

# Inicia os pesos das sinapses (centros de classe)
W = np.zeros([K,4])
W[0,:] = iris['data'][0  ,0:4]
W[1,:] = iris['data'][50 ,0:4]
W[2,:] = iris['data'][100,0:4]

x = np.zeros([1,4])

# Fase de treinamento.
for i in range(numEpocas):
    for j in range(Meio_N):
        # Apresentando um padrão de treinamento a rede.
        d = np.zeros([3,1]);
        #x = np.copy(iris['data'][Treinamento[j],0:4])
        x = iris['data'][Treinamento[j],0:4]
        for l in range(K):
            d[l] = np.linalg.norm(x - W[l])
            
        # Primeiro critério de pertinência.
        #dist, winner = min(d)
        dist, winner = min((d[p],p) for p in range(len(d))) 
        
        # Segundo critério de pertinência.
        if dist < r:
            # Neurônio vencedor treina!
            W[winner,:] = W[winner,:] + alfa*(x - W[winner,:])
            
# Fase de testes!
for i in range(Meio_N):
    x = iris['data'][Teste[i], 0:4]
    for l in range(K):
        d[l] = np.linalg.norm(x - W[l,:])
        
    # Primeiro critério de Pertinência - Menor distância.
    dist, winner = min((d[p],p) for p in range(len(d)))

    # Segundo critério de pertinência
    if dist < r:
        Classificacao[i] = winner;    
    else:
        Classificacao[i] = 0;
  
    Classificacao_Original[i] = iris['target'][Teste[i]]

erro = (np.count_nonzero(Classificacao - Classificacao_Original)/75.0)*100
print(erro)
    
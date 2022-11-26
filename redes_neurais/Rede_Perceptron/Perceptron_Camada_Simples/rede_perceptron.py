import numpy as np

# Define o número de épocas e o número de amostras (q)

numEpocas = 70000
q = 6

# Atributos

peso = np.array([113,122,107,98,115,120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])

# Bias
bias = 1

# Entrada do perceptron
X = np.vstack((peso, pH))
y = np.array([-1, 1, -1, -1, 1, 1])

# Taxa de aprendizado
eta = 0.1

# Define o vetor de pesos
W = np.zeros([1, 3])    # Duas entradas + o bias

# Array para armazenar os erros
e = np.zeros(6)


def funcaoAtivacao(valor):
    # A função de ativação a degrau bipolar
    if valor < 0.0:
        return(-1)
    else:
        return(1)
    
for j in range(numEpocas):
    for k in range(q):
        # Insere o bias no vetor de entrada
        Xb = np.hstack((bias, X[:,k]))
        
        # Calcula o campo induzido
        V = np.dot(W, Xb)        # Equação(5)
        
        # Calcula a saída a perceptron
        yr = funcaoAtivacao(V)   # Equação(6)
        
        # Calcula o erro: e = (y - yr)
        e[k] = y[k]- yr
        
        # Treinamento do perceptron
        W = W + eta*e[k]*Xb

print("Vetor de erros (e) = " + str(e))       
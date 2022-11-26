## Introdução a rede Perceptron

- Neste código criamos uma representação simples da rede, com 2 arrays e utilizando somente a biblioteca numpy que possibilita o calculo de vetores e matrizes com uma melhor performance.

#### __Problema a ser resolvido__
- Temos que prever se determinada fruta é uma macã ou uma laranja.
- Vamos utilizar 2 features como peso e pH

#### __Conclusões__

- Modelo com  70000 epocas, bias igual a 1 e taxa de aprendizado igual a 0.1 conseguiu aprender bem os valores de entrada na rede.
- Vetor de erros = [0,0,0,0,0,0]

- Porém valores com 63000 epocas teve dificuldade, e não conseguiu generalizar em 2 valores.

- Vetor de erros =  [ 0.  2.  0. -2.  0.  0.]


#### é necessário sempre ajustar os valores

# Projetos

# 1) Modelo de Regressão Linear via Gradiente Descendente Estocástico com Mini-lotes

Links dos artigos associados do Medium: 

- https://medium.com/@samir.silva12342/regress%C3%A3o-linear-com-gradiente-descendente-estoc%C3%A1stico-com-mini-lotes-857818864b6a

- https://medium.com/@samir.silva12342/analisando-modelos-simples-e-complexos-de-redes-neurais-aplicados-%C3%A0-regress%C3%A3o-linear-c0b021ecbe0c

Há quatro notebooks na pasta `linear-regression`:

- `01-model-implementation-standard.ipynb`: contém a implementação passo-a-passo do modelo de regressão linear utilizando Gradiente Descendente Estocástico com Mini-lotes.
- `02-validation.ipynb`: contém uma aplicação prática do modelo, apresentando também as funções utilizadas para dividir os dados em training, validation e test.
- `03-model-implementation-L2regularization.ipynb`: contém a implementação passo-a-passo do modelo utilizando também minimização do structural risk e regularização de complexidade via fórmula de regularização L2.
- `04-linear-regression-neural-network.ipynb`: contém a análise de modelos de redes neurais aplicados ao mesmo dataset usado nos outros notebooks.
 
A regressão linear é um dos modelos mais conhecidos no contexto do aprendizado de máquina, resumindo-se à adequação de um modelo linear a dados cujos preditores (features) são altamente correlacionados com o valor que se quer estimar (target). Ela pode ser implementada através do algoritmo de Gradiente Descendente Estocástico: um algoritmo iterativo de otimização dos parâmetros da função de regressão baseado na minimização da função de custo (ou de perda). Essa minimização é feita com o cálculo do gradiente negativo da função de custo em relação aos parâmetros da função de regressão (weights e bias).

Os notebooks desenvolvem passo-a-passo o modelo de regressão linear seguindo uma lógica para aprendizado de sua aplicação e implementação, seguindo as boas práticas utilizadas no aprendizado de máquina.

## Dados e Definições (disponíveis também no notebook)

![alt text](https://github.com/Samirnunes/machine-learning-implementations/blob/main/linear-regression/images/dados_e_definicoes.PNG)

## Tecnologias e Bibliotecas Utilizadas

- Jupyter Notebook
- Python
- Pandas
- Numpy
- Matplotlib

# 2) Regressão Logística: uma aplicação na detecção de pacientes de alto risco da Covid-19

Link do artigo associado do Medium: https://medium.com/@samir.silva12342/implementa%C3%A7%C3%A3o-do-zero-regress%C3%A3o-log%C3%ADstica-3e0f4dff9c32
 
Há dois notebooks na pasta `logistic-regression`:

- `01-logistic-regression-sklearn.ipynb`: nele, realizei a aplicação passo-a-passo das boas práticas de feature engineering no dataset 'Covid Data', disponível no Kaggle e apliquei um modelo de regressão logística aos dados para prever pacientes de alto risco de óbito devido à Covid-19, utilizando a biblioteca 'scikit-learn', do Python.

- `02-logistic-regression-implementation.ipynb`: neste notebook, realizei, passo-a-passo, a formulação matemática e a implementação em Python de um modelo de regressão logística utilizando regularização L1.

A regressão logística é um dos modelos de Machine Learning mais conhecidos e é aplicada quando queremos prever classificações através de probabilidades. Seu princípio envolve a utilização da função de ativação sigmoide na soma ponderada dos valores das features para obtenção de probabilidades que indicarão, ao serem comparadas com um threshold (limiar), qual a classificação final do target naquela situação.

## Tecnologias e Bibliotecas Utilizadas

- Jupyter Notebook
- Python
- Pandas
- Sklearn
- Numpy
- Matplotlib

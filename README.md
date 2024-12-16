English description below

Classificação e Regressão com Técnicas de Boosting

Este repositório contém três implementações de técnicas de boosting em Python, aplicadas em diferentes tipos de dados e cenários. 
O foco está em comparar os resultados de três algoritmos: 

AdaBoost, 

Gradient Boosting e 

Histogram-Based Gradient Boosting. 

Cada arquivo contém uma implementação distinta de cada algoritmo, usando o scikit-learn para AdaBoost e Gradient Boosting, e o HistGradientBoosting.

Arquivos

1. AdaBoost
   
Neste arquivo, eu implemento o AdaBoostClassifier usando classificadores baseados em árvores de decisão, com diferentes configurações:

Decision Stumps (árvores de decisão rasas, max_depth=1)

Árvores de decisão mais profundas (com max_depth=5)

Taxa de aprendizado alta (com learning_rate=1.0)

SAMME.R, o algoritmo de boosting Real AdaBoost, que utiliza probabilidades de classe.

O objetivo é explorar a performance do AdaBoost com diferentes parâmetros e observar como o modelo responde às mudanças nas configurações.

2. Gradient Boosting
   
No segundo arquivo, eu implemento o Gradient Boosting com regressão e classificação, 
comparando a performance entre um modelo manual e um modelo com o GradientBoostingRegressor do scikit-learn:

Gradient Boosting com parâmetros básicos (com max_depth=2, learning_rate=1.0 e 500 estimadores)

Gradient Boosting com taxa de aprendizado menor (com learning_rate=0.05 e 500 estimadores)

Gradient Boosting estocástico com amostragem (subsample=0.25)

Além disso, também é mostrado o desempenho do modelo em tarefas de classificação e regressão, utilizando diferentes métricas de avaliação, como erro médio absoluto (MAE) e R².

3. Histogram-Based Gradient Boosting
   
No terceiro arquivo, é implementado o Histogram-Based Gradient Boosting (HGB), que é uma versão mais eficiente do Gradient Boosting para grandes volumes de dados.

Aqui, eu combino dados contínuos e categóricos, utilizando o HistGradientBoostingRegressor para realizar previsões.

O modelo é comparado com outros modelos, como Gradient Boosting, XGBoost, LightGBM e CatBoost, para avaliar o desempenho em termos de precisão e eficiência:

Uso de um Pipeline para pré-processar dados categóricos com OneHotEncoder e aplicar o HGB.

O modelo é avaliado utilizando o erro médio absoluto (MAE) e o R².


Resultados

AdaBoost

Decision Stumps (max_depth=1): 

O modelo teve um desempenho não muito razoável, com uma accuracia de aproximadamente 72,06%, mas sofreu com a variação dos dados.

Árvores de decisão profundas (max_depth=5): 

O desempenho foi superior, com accuracia de cerca de 89,41%, demonstrando a eficácia de árvores mais profundas no aumento da capacidade de modelagem do AdaBoost.

Taxa de aprendizado alta (learning_rate=1.0): 

Aumentou o risco de overfitting e entregou uma accuracia de 73,38%.

SAMME.R: 

Resultados muito próximos do modelo com taxa de aprendizado alta, com uma accuracia de cerca de 73,38%.

Gradient Boosting

Gradient Boosting (max_depth=2, learning_rate=1.0, 500 estimadores): 

Erro Absoluto Médio (MAE): 1.4837e-16
R² Score: 1.0

Gradient Boosting (learning_rate=0.05, 500 estimadores): 

Erro Absoluto Médio (MAE): 135.0401
R² Score: 0.0715

Gradient Boosting Estocástico (subsample=0.25): 

Erro Absoluto Médio (MAE): 57.5334
R² Score: 0.6044

Histogram-Based Gradient Boosting

O Histogram-Based Gradient Boosting (HGB) apresentou resultados notáveis em termos de eficiência para grandes volumes de dados. 
Ao ser comparado com outros modelos como XGBoost, LightGBM e CatBoost, o HGB demonstrou ser competitivo:

Histogram-Based Gradient Boosting Regressor (Contínuos e Categóricos):

Mean Absolute Error (MAE): 0.08919953272434572
R² Score: 0.9976214780923666

Histogram-Based Gradient Boosting Regressor (Contínuos):

HGB: R² de 0.9976 e MAE de 0.0893.

XGBoost: R² de 0.9977 e MAE de 0.0874.

LightGBM: R² de 0.9976 e MAE de 0.0898.

CatBoost: R² de 0.9976 e MAE de 0.0905.







Classification and Regression with Boosting Techniques

This repository contains three implementations of boosting techniques in Python, applied to different types of data and scenarios. 
The focus is on comparing the results of three algorithms:

AdaBoost

Gradient Boosting

Histogram-Based Gradient Boosting

Each file contains a distinct implementation of each algorithm, using scikit-learn for AdaBoost and Gradient Boosting, and HistGradientBoosting.

Files

1. AdaBoost
   
In this file, I implement the AdaBoostClassifier using decision tree-based classifiers with different configurations:

Decision Stumps (shallow decision trees, max_depth=1)

Deeper Decision Trees (max_depth=5)

High Learning Rate (learning_rate=1.0)

SAMME.R, the Real AdaBoost boosting algorithm, which uses class probabilities.

The goal is to explore AdaBoost’s performance with different parameters and observe how the model responds to changes in configurations.

2. Gradient Boosting
   
In the second file, I implement Gradient Boosting for both regression and classification, 
comparing the performance between a manual model and a model using the GradientBoostingRegressor from scikit-learn:

Gradient Boosting with basic parameters (max_depth=2, learning_rate=1.0, 500 estimators)

Gradient Boosting with a smaller learning rate (learning_rate=0.05, 500 estimators)

Stochastic Gradient Boosting with subsampling (subsample=0.25)

Additionally, the model’s performance in classification and regression tasks is shown, using different evaluation metrics such as Mean Absolute Error (MAE) and R².

3. Histogram-Based Gradient Boosting
   
In the third file, Histogram-Based Gradient Boosting (HGB) is implemented, which is a more efficient version of Gradient Boosting for large datasets.

Here, I combine continuous and categorical data, using the HistGradientBoostingRegressor to make predictions.

The model is compared with others, such as Gradient Boosting, XGBoost, LightGBM, and CatBoost, to assess performance in terms of accuracy and efficiency:

A Pipeline is used to preprocess categorical data with OneHotEncoder and apply HGB.

The model is evaluated using Mean Absolute Error (MAE) and R².

Results

AdaBoost

Decision Stumps (max_depth=1):

The model performed reasonably, with an accuracy of approximately 72.06%, but suffered from data variation.

Deeper Decision Trees (max_depth=5):

The performance was better, with an accuracy of about 89.41%, demonstrating the effectiveness of deeper trees in boosting AdaBoost’s modeling capacity.

High Learning Rate (learning_rate=1.0):

Increased the risk of overfitting, with an accuracy of 73.38%.

SAMME.R:

Results were very similar to the high learning rate model, with an accuracy of around 73.38%.

Gradient Boosting

Gradient Boosting (max_depth=2, learning_rate=1.0, 500 estimators):

Mean Absolute Error (MAE): 1.4837e-16

R² Score: 1.0

Gradient Boosting (learning_rate=0.05, 500 estimators):

Mean Absolute Error (MAE): 135.0401

R² Score: 0.0715

Stochastic Gradient Boosting (subsample=0.25):

Mean Absolute Error (MAE): 57.5334

R² Score: 0.6044

Histogram-Based Gradient Boosting

Histogram-Based Gradient Boosting (HGB) showed remarkable results in terms of efficiency for large datasets. 
When compared to other models like XGBoost, LightGBM, and CatBoost, HGB proved competitive:

Histogram-Based Gradient Boosting Regressor (Continuous and Categorical):

Mean Absolute Error (MAE): 0.0892

R² Score: 0.9976

Histogram-Based Gradient Boosting Regressor (Continuous):

HGB:

R²: 0.9976

MAE: 0.0893

XGBoost:

R²: 0.9977

MAE: 0.0874

LightGBM:

R²: 0.9976

MAE: 0.0898

CatBoost:

R²: 0.9976

MAE: 0.0905

# Model Implementation From Scratch

## 1) Linear Regression with Mini-Batches Stochastic Gradient Descent

- Article on Medium: https://medium.com/@samir.silva12342/implementação-de-modelos-do-zero-regressão-linear-b3cdb9de1a24
 
The linear regression is one of the most well-known models in the context of machine learning, involving fitting a linear model to data whose predictors (features) are highly correlated with the target value to be estimated. It can be implemented through the Stochastic Gradient Descent algorithm: an iterative optimization algorithm for the parameters of the regression function based on minimizing the cost (or loss) function. This minimization is achieved by calculating the negative gradient of the cost function with respect to the parameters of the regression function (weights and bias).

The codes step-by-step develop the linear regression model, following a logical progression for learning its application and implementation. They adhere to best practices commonly used in machine learning.

### Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/data.png" alt="Material Bread logo">
<p>

### Results

#### Algorithm

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/explanation1.png" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/explanation2.png" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/explanation3.png" alt="Material Bread logo">
<p>

#### Images

- Correlations Heatmap

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/correlations_heatmap.png" alt="Material Bread logo">
<p>

- Model's Weights X Regularization Parameter

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/weights_lambda.png" alt="Material Bread logo">
<p>

- Train Loss

 <p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/train_loss.png" alt="Material Bread logo">
<p>

- Final Test Loss

<p align="center">
    <img width="400" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/linear_regression/images/final_loss.png" alt="Material Bread logo">
<p>

### Technologies and Libraries

- Jupyter Notebook
- Python
- Pandas
- Numpy
- Matplotlib

## 2) Logistic Regression with Mini-Batches Stochastic Gradient Descent

- Article on Medium: https://medium.com/@samir.silva12342/model-implementation-from-scratch-logistic-regression-737ee80cba7d

Logistic regression is one of the most well-known Machine Learning models and is applied when we want to predict classifications through probabilities. Its principle involves the use of the sigmoid activation function in the weighted sum of feature values to obtain probabilities. These probabilities, when compared to a threshold, indicate the final classification of the target in that situation.

### Data

COVID-19 Dataset from Kaggle: https://www.kaggle.com/datasets/meirnizri/covid19-dataset?source=post_page-----737ee80cba7d--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/data.PNG" alt="Material Bread logo">
<p>


### Results

#### Algorithm

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/explanation1.PNG" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/explanation2.PNG" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/explanation3.PNG" alt="Material Bread logo">
<p>

#### Images

- Train Loss

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/train_loss.PNG" alt="Material Bread logo">
<p>

- Precision-Recall Plot
  
<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/precision_recall_plot.PNG" alt="Material Bread logo">
<p>

- Precision-Recall Scores

<p align="center">
    <img width="300" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/precision_recall_scores.PNG" alt="Material Bread logo">
<p>
 
- Prediction Count
 
<p align="center">
    <img width="300" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/prediction_count.PNG" alt="Material Bread logo">
<p>
 
- Final Loss

<p align="center">
    <img width="300" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/model_implementation/logistic_regression/images/final_loss.PNG" alt="Material Bread logo">
<p>
 
### Technologies and Libraries

- Jupyter Notebook
- Python
- Pandas
- Sklearn
- Numpy
- Matplotlib

## 3) K-means

IN PROGRESS...

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import random as rd

df = pd.read_csv('C:/Users/user/Downloads/python scripts/data.csv')


X = df['x']
Y = df['y']

x=X.to_numpy()
y=Y.to_numpy()
n=700

plt.scatter(X, Y, color='blue', label='Données originales')  # Tracer les données originales avec plt.scatter()

def get_prediction(model, x):
    alpha_hat = model['alpha']
    beta_hat = model['beta']
    
    return alpha_hat + beta_hat * x

test_model = {'alpha': 2, 'beta': 2}

predictions = get_prediction(test_model, X)

plt.scatter(X, predictions, color='orange', label='Prédictions')

plt.legend()  

def mean_square_error(y,y_predicton):
    return np.sum(np.square(y - y_predicton))/n

def mean_absolute_error(y,y_predicton):
    return np.sum(abs(y - y_predicton))/n

print(mean_square_error(Y,predictions))
print(mean_absolute_error(Y,predictions))

def get_best_model(x,y):
    x_bar = np.average(x)
    y_bar = np.average(y)

    top = np.sum((x - x_bar)*(y - y_bar))
    bot = np.sum((x - x_bar)**2)
    beta_hat = top / bot

    alpha_hat = y_bar - beta_hat*x_bar

    model = {'alpha':alpha_hat, 'beta':beta_hat}

    return model

best_model=get_best_model(x,y)
best_predictions=get_prediction(best_model,X)

plt.scatter(X,best_predictions,color='green',label='best predictions')

plt.show()


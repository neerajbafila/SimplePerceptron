import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib


class Perceptron:
    def __init__(self, learning_rate: float=None, epochs: int=None):
        self.weights = np.random.randn(3) * 1e-4 # initializing initial weights for OR gate input
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def _z_outcome(self, inputs, weights):
        return np.dot(inputs,weights)
    
    def activation_function(self, z_outcome):
        return np.where(z_outcome > 0, 1, 0)
    
    def fit(self, X, y):
        self.X = X
        self.y = y

        X_with_biase = np.c_[self.X, -np.ones((len(self.X), 1))]
        print("X_with_biase  :",X_with_biase)
        for epoch in range(self.epochs):
            print("*"*30)
            print(f'epoch no {epoch}')
            z = self._z_outcome(X_with_biase, self.weights)
            print(f'Z values after epoch {epoch} : {z}')
            y_hat = self.activation_function(z)
            print(f'y_hat values after epoch {epoch} : {y_hat}')
            self.error = self.y - y_hat
            print(f'Error values after epoch {epoch} \n: {self.error}')
            #weight update
            self.weights = self.weights + self.learning_rate * np.dot(X_with_biase.T, self.error)
            print(f'New Weights after epoch {epoch} : {self.weights}')

    def predict(self,X):
        X_with_biase = np.c_[X, -np.ones((len(X), 1))]
        z = self._z_outcome(X_with_biase, self.weights)
        return self.activation_function(z)
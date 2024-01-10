import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin


def encode(encoder_weights: list, encoder_biases: list, data: pd.DataFrame):
    """
    Decrease data dimensionality by transforming input into values from hidden layer within the autoencoder.
    :param encoder_weights: list
    :param encoder_biases: list
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    data_ = data.copy()

    # calculate values of each record for each layer within the encoder part
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index + 1 == len(encoder_weights):
            # if we are at the final hidden layer apply simple linear model
            data_ = data_ @ w + b
        else:
            # otherwise, apply the relu function
            data_ = np.maximum(0, data_ @ w + b)

    return data_

class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, nn_params, n_hidden_layer):
        self.nn_params = nn_params
        self.n_hidden_layer = n_hidden_layer
        self.autoencoder = MLPRegressor(alpha=1e-15, hidden_layer_sizes=self.nn_params, random_state=17, max_iter=20000)

    def fit(self, X, y=None):
        self.autoencoder.fit(X, X)
        return self

    def transform(self, X, y=None):
        print("Estimate each user hidden unit values from encoder part of the autoencoder")
        w = self.autoencoder.coefs_
        biases = self.autoencoder.intercepts_
        encoder_weights = w[0:self.n_hidden_layer]
        encoder_biases = biases[0:self.n_hidden_layer]
        return encode(encoder_weights, encoder_biases, X)
# This file contains tests for the logistic regression implementation.

import numpy as np 
from Classic_ML import train_logistic_regression

def _sigmoid(z):
    """Numerically stable sigmoid implementation.
    
    args:
        z: A numpy array of any shape.
    
    returns:
        A numpy array of the same shape as z, where each element is the sigmoid of the corresponding element in z.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def test_logistic_regression():
    """
    Test the logistic regression implementation on a simple dataset.
    """
    # Create a simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Train the logistic regression model
    w, b = train_logistic_regression(X, y, lr=0.1, steps=1000)
    
    # Test the model on the training data
    z = X @ w + b 
    p = _sigmoid(z)
    predictions = (p >= 0.5).astype(int)
    
    # Assert that the predictions match the true labels
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

    
    
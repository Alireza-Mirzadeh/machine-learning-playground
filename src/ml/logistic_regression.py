import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation.
    
    args:
        z: A numpy array of any shape.
    
    returns:
        A numpy array of the same shape as z, where each element is the sigmoid of the corresponding element in z.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train a logistic regression model using gradient descent.
    
    args:
        X: A numpy array of shape (N, d) containing the training data, where N is the number of samples and d is the number of features.
        y: A numpy array of shape (N,) containing the binary labels (0 or 1) for each sample.
        lr: Learning rate for gradient descent (default: 0.1).
        steps: Number of iterations for gradient descent (default: 1000).
        
    returns:
        A tuple (w, b) where w is the weight vector of shape (d,) and b is the bias term (scalar).
    """
    
    # Number of samples and number of features
    N, d = X.shape

    # Initialize parameters

    # Weight vector
    w = np.zeros(d)
    # Bias term
    b = 0.0

    # Gradient Descent Loop
    for _ in range(steps):
        
        # Linear model
        z = X @ w + b # Shape: (N,)

        # Apply sigmoid to get the probability
        p = _sigmoid(z) # Shape: (N,)

        # Compute error between predicted probabilities and true labels 
        error = p - y 

        # Compute gradients 
        dw = (1 / N) * (X.T @ error)

        db = (1 / N) * np.sum(error)

        # Update parameters
        w = w - lr * dw 
        b = b - lr * db 

    return (w, b)

# Demo 
if __name__ == "__main__":
    
    # Sample dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Train logistic regression model
    w, b = train_logistic_regression(X, y)
    print("Weights:", w)
    print("Bias:", b)
    
    # Predict probabilities for the training data
    z = X @ w + b
    p = _sigmoid(z)
    print("Predicted probabilities:", p)
    
    # Predict class labels based on a threshold of 0.5
    predictions = (p >= 0.5).astype(int)
    print("Predicted class labels:", predictions)

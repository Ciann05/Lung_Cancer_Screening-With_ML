import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.theta = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.asarray(z, dtype=float)
        probabilities = np.empty_like(z, dtype=float)
        positive = z >= 0
        probabilities[positive] = 1 / (1 + np.exp(-z[positive]))
        negative = ~positive
        exp_z = np.exp(z[negative])
        probabilities[negative] = exp_z / (1 + exp_z)
        return probabilities

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        # Initialize model parameters
        self.theta = np.zeros(X.shape[1])
        self.bias = 0.0

        # SGD training
        for epoch in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):

            # Shuffle the training data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # Loop through mini-batches
            for start in range(0, len(X_shuffled), self.batch_size):

                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Calculate gradients
                theta_gradient, bias_gradient = self.gradient(
                    X_batch,
                    Y_batch)

                # Update parameters
                self.theta -= self.learning_rate * theta_gradient
                self.bias -= self.learning_rate * bias_gradient

        return self

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.

            Stability trick: dL/dz = p - y for BCE, so we never differentiate through log(p).
            L2 contributes lambda * theta to dL/dtheta (bias is not regularized).
        """
        # Calculate model scores
        z = np.dot(X, self.theta) + self.bias
        probabilities = self._sigmoid(z)

        error = probabilities - Y

        # Gradient with respect to theta
        theta_gradient = np.dot(X.T, error) / len(Y)
        # Add L2 regularization
        theta_gradient += self.regularization_lambda * self.theta
        # Gradient with respect to bias
        bias_gradient = np.mean(error)

        return theta_gradient, bias_gradient

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.theta + self.bias)

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

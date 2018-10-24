import numpy as np
import matplotlib.pyplot as plt
from tools import *


np.seterr(over='raise')


class Model:
    """Abstract class implementing methods common to all models"""

    def misclassification_error(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred != y) / y.shape[0]

    def plot_classification_boundary(self, X, y, c=0.01, step=500):
        plt.figure(figsize=(10, 10))
        self.fit(X, y)
        plot_data(X, y)
        plot_boundary(X, self.predict, step, alpha_c=0.1)
        plt.title(self.model_name)


class LDA(Model):
    """
    LDA model class
    """

    model_name = 'LDA'

    def fit(self, X, y):
        """
        Compute maximum likelihood estimators for a given dataset
        """

        # number of samples and number of positive samples to compute prior pi
        n = y.shape[0]
        Ny = np.sum(y == 1)

        self.pi = Ny / n

        # computation of the empiric mean and covariance
        self.mu0 = np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)

        sigma_0 = (X[y == 0] - self.mu0).T.dot((X[y == 0] - self.mu0)) / (n - Ny)
        sigma_1 = (X[y == 1] - self.mu1).T.dot((X[y == 1] - self.mu1)) / Ny

        self.sigma = Ny / n * (sigma_1) + (n - Ny) / n * (sigma_0)
        self.sigma_inv = np.linalg.inv(self.sigma)  # saving the inverse allows to avoid extra computations

    def predict_proba(self, X):
        """
        Infer the probability to belong to class 1 for each point in a given dataset
        """

        # this disjunction allows to predict the probability of a single or several samples
        if X.ndim != 1:
            logp1 = np.log(self.pi) \
                    - 0.5 * np.apply_along_axis(lambda x: (x - self.mu1).dot(self.sigma_inv).dot((x - self.mu1).T),
                                                arr=X,
                                                axis=1)
            logp0 = np.log(1 - self.pi) \
                    - 0.5 * np.apply_along_axis(lambda x: (x - self.mu0).dot(self.sigma_inv).dot((x - self.mu0).T),
                                                arr=X,
                                                axis=1)

            p1 = np.exp(logp1)
            p0 = np.exp(logp0)

            p1 = p1 / (p0 + p1)

        else:
            logp1 = np.log(self.pi) - 0.5 * (X - self.mu1).dot(self.sigma_inv).dot((X - self.mu1).T)
            logp0 = np.log((1 - self.pi)) - 0.5 * (X - self.mu0).dot(self.sigma_inv).dot((X - self.mu0).T)

            p1 = np.exp(logp1)
            p0 = np.exp(logp0)

            logp1 = p1 / (p0 + p1)

        return p1

    def predict(self, X):
        """
        Classify each point in a given dataset according to the infered probability to belong to class 1
        """

        return self.predict_proba(X) > 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-np.maximum(x, -100))) # prevent overflow


class LogisticRegression(Model):
    """
    Logistic regression model class
    """

    model_name = 'Logistic regression'

    def fit(self, X, Y, eps=1e-5):
        """
        Compute the optimal weights for a given dataset
        """
        
        # add an offset for the bias term
        offset = np.ones(X.shape[0])
        offset = offset.reshape(-1, 1)
        X = np.hstack((offset, X)) 

        p = X.shape[1]
        w = np.zeros(p)

        # iterative reweighted least squares
        convergence = False
        while not convergence:
            eta = sigmoid(X.dot(w))
            grad_w = np.dot(X.T, Y - eta)
            h_w = - X.T.dot(np.diag(eta * (1 - eta))).dot(X)

            convergence = (- grad_w.T.dot(np.linalg.inv(h_w)).dot(grad_w) / 2 < eps)
            if convergence:
                break
            else:
                w = w - np.dot(np.linalg.inv(h_w), grad_w)

        self.w = w
        return w

    def predict_proba(self, X):
        """
        Compute the probability to belong to class 1 for each point in a given dataset
        """
        
        # add an offset for the bias term
        offset = np.ones(X.shape[0])
        offset = offset.reshape(-1, 1)
        X = np.hstack((offset, X))

        return sigmoid(np.dot(X, self.w))

    def predict(self, X):
        """
        Classify each point in a given dataset
        """
        
        return self.predict_proba(X) > 0.5


class LinearRegression(Model):
    """
    Linear regression model class
    """
    
    model_name = 'Linear regression'
    
    def fit(self, X, Y):
        """
        Compute the optimal weights for a given dataset
        """
        
        # add an offset for the bias term
        offset = np.ones(X.shape[0])
        offset = offset.reshape(-1, 1)
        X = np.hstack((offset, X))
        
        w =  np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        self.w = w
        
        return w

    def predict_proba(self,X, w=None):
        """
        Compute the scalar on which the classification decision is based
        """
        
        if w is None:
            w = self.w
            
        # add an offset for the bias term
        offset = np.ones(X.shape[0])
        offset = offset.reshape(-1, 1)
        X = np.hstack((offset, X))
        
        return X.dot(w)

    def predict(self,X, w=None):
        """
        Classify each point in a given dataset
        """
        
        if w is None:
            w = self.w
        
        return self.predict_proba(X,w) > 0.5


class QDA(Model):
    """
    QDA model class
    """
    
    model_name = 'QDA'

    def fit(self, X, y):
        """
        Compute maximum likelihood estimators for a given dataset
        """
        
        # number of samples (total and positive ones)
        n = y.shape[0]
        Ny = np.sum(y == 1)

        self.pi = Ny / n

        # compute empiric mean and covariance
        self.mu0 = np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)

        self.sigma_0 = (X[y == 0] - self.mu0).T.dot((X[y == 0] - self.mu0)) / (n - Ny)
        self.sigma_1 = (X[y == 1] - self.mu1).T.dot((X[y == 1] - self.mu1)) / Ny

        # save inverse matrices and det
        self.sigma0_inv = np.linalg.inv(self.sigma_0)
        self.sigma1_inv = np.linalg.inv(self.sigma_1)

        self.sigma0_det = np.linalg.det(self.sigma_0)
        self.sigma1_det = np.linalg.det(self.sigma_1)

    def predict_proba(self, X):
        """
        Infer the probability to belong to class 1 for each point in a given dataset
        """
        
        # this disjunction allows to predict the probability of a single or several samples
        if X.ndim != 1:
            logp1 = np.log(self.pi) - 0.5 * np.log(self.sigma1_det) \
                    - 0.5 * np.apply_along_axis(lambda x: (x - self.mu1).dot(self.sigma1_inv).dot((x - self.mu1).T),
                                                arr=X, 
                                                axis=1)
            
            logp0 = np.log(1 - self.pi) - 0.5 * np.log(self.sigma0_det) \
                    - 0.5 * np.apply_along_axis(lambda x: (x - self.mu0).dot(self.sigma0_inv).dot((x - self.mu0).T),
                                                arr=X,
                                                axis=1)

            p1 = np.exp(logp1)
            p0 = np.exp(logp0)

            p1 = p1 / (p0 + p1)
            
        else:
            logp1 = np.log(self.pi) - 0.5 * np.log(self.sigma1_det) \
                    - 0.5 * (X - self.mu1).dot(self.sigma1_inv).dot((X - self.mu1).T)
            
            logp0 = np.log((1 - self.pi)) - 0.5 * np.log(self.sigma0_det) \
                    - 0.5 * (X - self.mu0).dot(self.sigma0_inv).dot((X - self.mu0).T)

            p1 = np.exp(logp1)
            p0 = np.exp(logp0)

            logp1 = p1 / (p0 + p1)

        return p1

    def predict(self, X):
        """
        Classify each point in a given dataset
        """
        
        return self.predict_proba(X) > 0.5

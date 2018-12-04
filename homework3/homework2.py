import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix


def Jfunc(X, mu, z, k):
    J = 0
    for i in range(k):
         J += np.sum(np.power(np.linalg.norm(X[z==i] - mu[i,:], axis=1), 2))
    return J


def kmeans(X, k, eps):
    
    # initialization
    mu = X[np.random.randint(0, X.shape[0],k), :]
    convergence = False
          
    j = np.inf

    while not(convergence):
        # optimize over z
        z = np.argmin(distance_matrix(X, mu), axis=1)
        
        #optimize over mu
        for i in range(0,k):
            mu[i, :] = X[z==i].mean(axis=0)
            
        #convergence evaluation
        if np.abs(j - Jfunc(X, mu, z, k)) < eps:
            convergence = True
        else:
            j = Jfunc(X, mu, z, k)
    
    return mu, z


class GaussianMixture():
    
    def __init__(self, X, k, covariance_mode, eps=1e-3, n_itermax=1000):
        
        # input parameters
        self.X = X
        self.k = k
        self.covariance_mode = covariance_mode
        self.eps = eps  # convergence threshold
        self.n_itermax = n_itermax 
        
        
        # kmeans initialization
        mu, tau = kmeans(X, k, 0.1)
       
        self.tau = np.zeros(pd.get_dummies(tau).values.shape)

        # Gaussians mean and covariance matrix initialization
        self.mus = mu.T

        if covariance_mode == 'isotrope':
            self.sigmas = np.ones(k)
        else:
            self.sigmas = np.ones((k, self.X.shape[1], self.X.shape[1]))
            for j in range(self.k):
                self.sigmas[j] = np.eye(self.X.shape[1])
                
        self.pis = np.ones(k)/k
        
    def predict(self, X):
        self.X = X
        
        for j in range(self.k):
                self.tau[:, j] = self.pis[j] * multivariate_normal.pdf(X, 
                                                                   mean=self.mus[:, j], 
                                                                   cov=self.sigmas[j])
        self.tau = self.tau / self.tau.sum(axis=1).reshape(-1, 1)
        return np.argmax(self.tau, axis=1)
            
    def fit(self,verbose=False):
        loglikold = -np.inf
        for i in range(self.n_itermax):
            for j in range(self.k):
                self.tau[:, j] = self.pis[j] * multivariate_normal.pdf(self.X, 
                                                                   mean=self.mus[:, j], 
                                                                   cov=self.sigmas[j])
            self.tau = self.tau / self.tau.sum(axis=1).reshape(-1, 1)
            
            loglik = 0
            for j in range(self.k):
                log_pdf = multivariate_normal.logpdf(self.X, mean=self.mus[:, j], cov=self.sigmas[j])
                loglik += (self.tau[:,j] * log_pdf).sum() + self.tau[:,j].sum() * np.log(self.pis[j])
                loglik -= (self.tau[:,j] * np.log(self.tau[:,j])).sum()
            if verbose:
                print(loglik)
            if loglik < loglikold - self.eps:
                print('error')
            if loglik < loglikold + self.eps:
                break
            
            loglikold = loglik;

            self.pis = self.tau.mean(axis=0)
            for j in range(self.k):
                self.mus[:,j] = (self.X * (self.tau[:, j].reshape(-1, 1))).sum(axis=0) / self.tau[:,j].sum()

            if self.covariance_mode == 'isotrope':
                for j in range(self.k):
                    X_centered_norm = np.power(np.linalg.norm(self.X - self.mus[:, j], axis=1),2)
                    self.sigmas[j] = np.average(X_centered_norm, weights=self.tau[:, j])/2
            else:
                for j in range(self.k):
                    X_centered = self.X - self.mus[:,j]
                    sigma_i = (X_centered.T.dot(np.diag(self.tau[:, j]))).dot(X_centered)
                    sigma_i = sigma_i / np.diag(self.tau[:, j]).sum()
                    self.sigmas[j] = sigma_i
                
    def jfunc(self):
        loglik = 0
        
        for j in range(self.k):
            log_pdf = multivariate_normal.logpdf(self.X, mean=self.mus[:, j], cov=self.sigmas[j])
            loglik += (self.tau[:,j]* log_pdf).sum() + self.tau[:,j].sum() * np.log(self.pis[j])
            loglik -= (self.tau[:,j]*np.log(self.tau[:,j])).sum()
            
        return loglik

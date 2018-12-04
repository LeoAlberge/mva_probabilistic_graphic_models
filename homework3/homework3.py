import numpy as np


def log_sum_exp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def gaussian_log_pdf(x, mu, sigma):
    n = mu.size
    return - 0.5 * ((x - mu).dot(np.linalg.inv(sigma).dot(x - mu)) + n * np.log(2 * np.pi) + np.log(np.linalg.det(sigma)))


class GaussianHMM:
    """
    HMM class for gaussian emissions
    """
    
    def __init__(self, a, pi, mu, sigma, min_float=1e-50):
        assert(a.shape[0] == a.shape[1] == pi.shape[0] == mu.shape[0] == sigma.shape[0])
        self.num_states = a.shape[0]
        
        self.mu = mu
        self.sigma = sigma
        
        self.min_float = min_float
        self.log_a = np.log(a + min_float)
        self.log_pi = np.log(pi + min_float)
        
    # emission probabilities p(u_t|q_t=i)
    def log_f(self, u, i):
        return gaussian_log_pdf(u, self.mu[i], self.sigma[i])
    
    def log_likelihood(self, u):
        log_alpha, _, _, _ = self.forward_backward(u)
        return log_sum_exp(log_alpha[-1])
        
    def forward_backward(self, u):
        num_obs = u.shape[0]
        log_alpha = np.zeros((num_obs, self.num_states))
        log_beta = np.zeros((num_obs, self.num_states))
        
        # alpha recursion
        for t in range(num_obs):
            for i in range(self.num_states):
                if t == 0:
                    log_alpha[t, i] = self.log_pi[i] + self.log_f(u[0], i)
                else:
                    log_alpha[t, i] = log_sum_exp(log_alpha[t - 1, :] + self.log_a[:, i]) + self.log_f(u[t], i)
                
        # beta recursion
        for t in reversed(range(num_obs)):
            for i in range(self.num_states):
                if t == num_obs - 1:
                    log_beta[t, i] = 0.0
                else:
                    # array of p(u_t+1|q_t+1=j) for all j
                    p_emissions = np.array([self.log_f(u[t + 1], j) for j in range(self.num_states)])
                    
                    log_beta[t, i] = log_sum_exp(log_beta[t + 1, :] + self.log_a[i, :] + p_emissions)
                    
        # gamma and xi computation
        log_gamma = log_alpha + log_beta
        log_xi = np.zeros((num_obs - 1, self.num_states, self.num_states))
        
        for t in range(num_obs - 1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_xi[t, i, j] = log_alpha[t, i] + self.log_a[i, j] + log_beta[t + 1, j] + self.log_f(u[t + 1], j)
            
            log_gamma[t] -= log_sum_exp(log_alpha[t] + log_beta[t])
            log_xi[t] -= log_sum_exp(np.array([log_sum_exp(log_xi[t, i]) for i in range(self.num_states)]))
        
        return log_alpha, log_beta, log_gamma, log_xi
    
    def learn_parameters(self, u, tol, max_iter, verbose=False):
        """
        Expectation Maximization algorithm for HMM
        """
        
        num_obs = u.shape[0]
        
        former_log_likelihood = - np.inf
        for step in range(max_iter):
            
            # E step
            
            log_alpha, log_beta, log_gamma, log_xi = self.forward_backward(u)
            gamma = np.exp(log_gamma)
            xi = np.exp(log_xi)
            
            log_likelihood = log_sum_exp(log_alpha[-1])
            if verbose:
                print('step: {}, log likelihood: {}'.format(step, log_likelihood))
            
            # convergence check
            assert(former_log_likelihood <= log_likelihood)
            if abs(log_likelihood - former_log_likelihood) < tol:
                break
                
            former_log_likelihood = log_likelihood
            
            # M step
            
            # initial probability update
            self.log_pi = log_alpha[0] + log_beta[0]
            self.log_pi -= log_sum_exp(self.log_pi)
            
            # transition probabilities update
            for i in range(self.num_states):
                for j in range(self.num_states):
                    self.log_a[i, j] = log_sum_exp(log_xi[:, i, j]) - log_sum_exp(log_gamma[:-1, i])

            # means update
            for i in range(self.num_states):
                self.mu[i] = np.sum(gamma[:, i, np.newaxis] * u, axis=0) / np.sum(gamma[:, i])

           # covariance matrices update
            for i in range(self.num_states):
                cov_array = np.array([gamma[t, i] * np.outer(u[t] - self.mu[i], u[t] - self.mu[i]) 
                                      for t in range(num_obs)])
                self.sigma[i] = np.sum(cov_array, axis=0) / np.sum(gamma[:, i])
                    
    def decode(self, u):
        """
        Viterbi algorithm for decoding
        """
        
        num_obs = u.shape[0]
        log_v = np.zeros((num_obs, self.num_states))
        s = np.zeros(num_obs)
        
        for i in range(self.num_states):
            log_v[0, i] = self.log_pi[i] + self.log_f(u[0], i)
            
        for t in range(1, num_obs):
            for i in range(self.num_states):
                best_j = np.argmax(log_v[t - 1] + self.log_a[:, i])
                log_v[t, i] = log_v[t - 1, best_j] + self.log_a[best_j, i] + self.log_f(u[t], i)
                s[t - 1] = best_j
                
        s[num_obs - 1] = np.argmax(log_v[t - 1])
        
        return s
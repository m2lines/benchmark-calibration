import numpy as np
from numpy.linalg import norm

def norm_per_element(x):
    return np.sqrt(np.mean(x**2))

class InverseProblem:
    """
    A base class for inverse problems of the form
    y = G(u)
    """     
    def observation(self):
        '''
        This is an observation value which we want to match
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def _forward_map(self, u):
        '''
        This is a possibly noisy forward map G(u)
        '''
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _true_parameter(self):
        '''
        This is a true parameter u_{true} which we want to recover
        '''
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _error(self, u):
        '''
        This is the error in the solution of the inverse problem,
        i.e. ||u-u_{true}||_2 in simple case
        '''
        return norm_per_element(u - self._true_parameter())

    def forward_map(self, u):
        '''
        If u and y are matrices, then the leftmost dimension is considered to be 
        the ensemble size
        '''
        u = np.array(u)
        if u.ndim == 2:
            out = np.array([self._forward_map(u_i) for u_i in u])
            if out.ndim == 1:
                out = out.reshape(-1,1)
            return out
        else:
            raise ValueError("Input array must be 2D.")

class min_quadratic_function(InverseProblem):
    """
    A class which creates an inverse problem of the form
    0 = sum_{i=1}^n (u_i-a_i)^2
    with unique solution u = a
    """
    def __init__(self, n=2):
        self.dim_of_parameters = n
        self.dim_of_observations = 1
        self.a = np.random.rand(n)
    
    def observation(self):
        return 0.

    def _forward_map(self, u):
        return ((u-self.a)**2).sum()
    
    def _true_parameter(self):
        return self.a
    
class transcendental_function(InverseProblem):
    """
    A class which creates an inverse problem of the form
    0 = exp(-u) - u
    with unique solution u = 0.56714329
    """
    def __init__(self, n=2):
        self.dim_of_parameters = n
        self.dim_of_observations = n
    
    def observation(self):
        return np.zeros(self.dim_of_observations)

    def _forward_map(self, u):
        return np.exp(-u) - u
    
    def _true_parameter(self):
        return np.ones(self.dim_of_parameters) * 0.56714329
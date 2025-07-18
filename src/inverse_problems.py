import numpy as np
from numpy.linalg import norm

class InverseProblem:
    """
    A base class for inverse problems of the form
    y = G(u)
    """     
    def __init__(self):
        '''
        The dimensionality of parameters and observations
        must be defined in subclasses.
        '''
        self.dim_of_parameters = None
        self.dim_of_observations = None

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
    
    def _error(self, u):
        '''
        This is the error in the solution of the inverse problem,
        i.e. ||u-u_{true}||_2 in simple case
        '''
        raise NotImplementedError("Subclasses should implement this method.")

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

    def error(self, u):
        '''
        If u and y are matrices, then the leftmost dimension is considered to be 
        the ensemble size
        '''
        u = np.array(u)
        if u.ndim == 2:
            out = np.array([self._error(u_i) for u_i in u])
            if out.ndim == 1:
                out = out.reshape(-1,1)
            return out
        else:
            raise ValueError("Input array must be 2D.")

class min_quadratic_function(InverseProblem):
    """
    A class which creates an inverse problem of the form
    0 = sum_{i=1}^n (u_i-a_i)^2 / N
    with unique solution u = a
    """
    def __init__(self, n=2, a=None):
        self.dim_of_parameters = n
        self.dim_of_observations = 1

        if a is None:
            self.a = np.random.rand(n)
        else:
            self.a = np.array(a)
    
    def observation(self):
        return 0.

    def _forward_map(self, u):
        return ((u-self.a)**2).mean()
    
    def _error(self, u):
        return np.sqrt(((u-self.a)**2).mean())

# class find_unit_sphere(InverseProblem):
#     """
#     A class which creates an inverse problem of the form
#     1/N = mean_{i=1}^n (u_i-a_i)^2
#     with multiple solutions lying in unit sphere with center at $u=a$
#     """
#     def __init__(self, n=2, a=None):
#         self.dim_of_parameters = n
#         self.dim_of_observations = 1

#         if a is None:
#             self.a = np.random.rand(n)
#         else:
#             self.a = np.array(a)
        
#     def observation(self):
#         return 1./self.dim_of_parameters
    
#     def _forward_map(self, u):
#         return ((u-self.a)**2).mean()
    
#     def _error(self, u):
#         '''
#         This is the distance to the unit sphere centered at a
#         '''
#         return np.abs(np.sqrt(((u-self.a)**2).mean())-1./self.dim_of_parameters)
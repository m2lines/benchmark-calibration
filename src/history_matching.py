import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class HistoryMatching:
    """
    Class for solving the inverse problem (i.e., calibration)
    using history matching techniques.

    History matching iteratively repeats two steps:
    * Fit Gaussian Process (GP) to approximate the forward map
    * Find the extreme values of the Implausibility Function, 
    which guides where in parameter space to evaluate the forward map next
    """

    def __init__(self, inverse_problem=None,                 
                 initial_ensemble_size=10, 
                 prior=np.random.randn,
                 emulator_sample_size=10000,
                 implausibility_threshold=3.0):
        '''
        We specify the number of parameters and observations in order to generate the initial ensemble
        and to know how many optimization criteria we have.

        History matching is initialized with sampling a few combinations of parameters
        randomly ("initial_ensemble_size" argument)
        from the prior distribution, which is given by "prior" argument.

        "emulator_sample_size" specifies how many samples to evaluate 
        using gaussian process regression (GP) in each iteration
        These samples will be used to find the extremal values of the Implausibility Function.
        For simplicity, we sample GP from the prior distribution.
        '''
        self.initial_ensemble_size = initial_ensemble_size
        self.implausibility_threshold = implausibility_threshold
        self.prior = prior
        self.inverse_problem = inverse_problem
        self.emulator_sample_size = emulator_sample_size

        # Vector of parameters which is updated over iterations
        # (N_samples,dim_of_parameters)
        self.parameters = None
        # Vector of evaliuations of the forward map corresponding to these parameters
        # (N_samples,dim_of_observations)
        self.evaluations = None

        # Collect statistics about errors and the number of parameters
        self.error_new = []
        self.error_forward = []
        self.error_gp = []
        self.error_IF = []
        self.forward_evaluations = []
    
    def iteration(self):
        """
        Performs one iteration of history matching.
        The action of the algorithm is to update a vector
        of parameters and evaluate in new point the forward model
        """

        # Sample a few combinations of parameters from the prior distribution
        if self.parameters is None:
             self.parameters = self.prior(self.initial_ensemble_size, self.inverse_problem.dim_of_parameters)
             self.evaluations = self.inverse_problem.forward_map(self.parameters)

        # Step 1. Fit Gaussian Process to predict each observational dimensionality independently
        X = self.parameters    # (N_samples,dim_of_parameters)
        Y = self.evaluations   # (N_samples,dim_of_observations)

        # Make Y 2D array if it is not yet
        if Y.ndim == 1:
            Y = Y.reshape(-1,1)

        gps = []
        for i in range(Y.shape[1]):
            gp = GaussianProcessRegressor()
            gp.fit(X, Y[:,i])
            gps.append(gp)

        # Sample many points from the prior distribution and evaluate 
        # using cheap GP emulator
        X_test = np.random.rand(self.emulator_sample_size, self.inverse_problem.dim_of_parameters)

        Y_mean = []
        Y_std = []
        for gp in gps:
            y_mean, y_std = gp.predict(X_test, return_std=True)
            Y_mean.append(y_mean)
            Y_std.append(y_std)
        Y_mean = np.stack(Y_mean, axis=1)
        Y_std = np.stack(Y_std, axis=1)

        # Step 3. Find the implausibility function
        # We maximize implausibility across all dimensions of observation
        # So we want to find regions which are plausible w.r.t. to all dimension of observations
        implausibility = (np.abs(Y_mean - self.inverse_problem.observation()) / Y_std).max(axis=1)

        # Step 4. Rule out implausible points
        plausible_indices = implausibility < self.implausibility_threshold

        # Note: We may have a lot of plausible points, and we need to choose one which is the best

        # Step 5. In plausible points, keep exploring the parameter space
        # by taking the point with highest uncertainty
        # Here, we consider local maximum among dimension of observations

        # We keep sampling in this points of parameter space
        idx_to_sample = np.argmax(np.where(plausible_indices, Y_std.max(axis=1), -np.inf))
        
        # Step 6. Evaluate forward model at new parameter
        new_parameter  = X_test[idx_to_sample].reshape(1, -1)
        new_evaluation = self.inverse_problem.forward_map(new_parameter).reshape(1, -1)
        
        self.parameters = np.vstack((self.parameters, new_parameter))
        self.evaluations = np.vstack((self.evaluations, new_evaluation))

        # We suggest four proxies which solve the inverse problem the best
        # First, we evaluate error for the parameter which we just found
        error_new = self.inverse_problem._error(new_parameter)

        # Second, we take the parameter which is already evaluated with the forward model
        # Which gives the best fit of observations
        idx = np.argmin(np.abs(self.evaluations - self.inverse_problem.observation()).max(axis=1))
        parameter_forward = self.parameters[idx]
        error_forward = self.inverse_problem._error(parameter_forward)

        # Third, we do the same among points predicted by GP
        idx = np.argmin(np.where(plausible_indices, np.abs(Y_mean-self.inverse_problem.observation()).max(axis=1), np.inf))
        parameter_gp = X_test[idx]
        error_gp = self.inverse_problem._error(parameter_gp)

        # Fourth, we find the minimum of plausibility function
        idx = np.argmin(implausibility)
        parameter_IF = X_test[idx]
        error_IF = self.inverse_problem._error(parameter_IF)

        # Collect statistics
        self.error_new.append(error_new)
        self.error_forward.append(error_forward)
        self.error_gp.append(error_gp)
        self.error_IF.append(error_IF)
        self.forward_evaluations.append(self.evaluations.shape[0])
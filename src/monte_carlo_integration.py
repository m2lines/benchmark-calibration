import numpy as np
def monte_carlo_integral(f, ens_size=None, niters=5, y=0, noise=0, seed=None, m=None, Sigma=None, asymptotic=False):
  '''
  This method iteratively evaluates the integral over the
  posterior distribution using Monte Carlo integration:
  int(f(u) rho_post(u), du) = sum f(u_i) w_i / sum w_i
  Weights w_i solely depend on the likelihood

  Replacing f(u) with u and u u^T, we obtain mean and covariance of the posterior distribution
  '''
  if ens_size is None:
    ens_size = f.dim_of_parameters * 10

  rng = np.random.default_rng(seed=seed)
  
  x_ens_data = []
  y_ens_data = []

  for i in range(niters):
    # x_ens.size = (dim_of_parameters, ens_size)
    x_ens = rng.multivariate_normal(mean=m.reshape(-1), cov=Sigma, size=ens_size).T
    y_ens = np.stack([f(x) for x in x_ens.T]).T

    x_ens_data.append(x_ens)
    y_ens_data.append(y_ens)

    # Compute likelihood (optionally, minus prior)
    # for each ensemble member individually
    Phi = 0.5 * np.sum((y_ens - y)**2, axis=0) / noise**2
    
    # Shift log-density, which is positive, to avoid overflow
    Phi -= Phi.min()

    if asymptotic:
      # Here we assume that the quadrature weights are too bad
      # because peak of posterior is too narrow to capture it on a given grid
      # We artificially increase the spread of the posterior as follows:
      # np.exp(-beta * Phi)
      # and consider extra small beta -> 0
      # In particular we chouse beta such that the minimal weights is 0
      # in linear appxoximation of Phi:
      # np.exp(-beta * Phi) = 1 - beta * Phi = 0
      # that is beta = 1 / Phi.max()
      Phi /= Phi.mean()
    
    print('Phi in range', Phi.min(), Phi.max())
    
    # Quadrature weights  
    w = np.exp(-Phi)

    #print('w.sum()=', w.sum(), 'Phi.sum()=', Phi.sum())
    #print(w)
    print('min/max/sum w=', np.min(w), np.max(w), np.sum(w))
    # Computing first two moments of posterior using quadrature weights
    m = (x_ens * w).sum(axis=-1, keepdims=True) / w.sum()
    Sigma = ((x_ens-m) * w) @ (x_ens-m).T / w.sum()

  return m.reshape(-1), dict(x_ens=x_ens_data, y_ens=y_ens_data)
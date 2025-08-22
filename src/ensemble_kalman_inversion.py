import numpy as np
def EKI(f, ens_size=None, niters=5, y=0, noise=0, seed=None, uniform=False, low=None, high=None, randomized_likelihood=False):
  '''
  Reproduces EnsembleKalmanProcesses.jl with parameters:

  Inversion()
  scheduler = DefaultScheduler(1)
  accelerator = DefaultAccelerator(),
  localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization()
  deterministic_forward_map=false
  '''
  if ens_size is None:
    ens_size = f.dim_of_parameters * 10

  rng = np.random.default_rng(seed=seed)
  
  if uniform:
    x_ens = np.stack([rng.uniform(low[i], high[i], ens_size) for i in range(len(low))])
  else:
    x_ens = rng.normal(size=(f.dim_of_parameters, ens_size))

  mean=lambda x: x.mean(axis=-1,keepdims=True)
  x_ens_data = []
  y_ens_data = []
    
  y_hat = y
  for i in range(niters):
    y_ens = np.stack([f(x) for x in x_ens.T]).T

    x_ens_data.append(x_ens)
    y_ens_data.append(y_ens)

    cov_xy = (x_ens-mean(x_ens)) @ (y_ens-mean(y_ens)).T / ens_size
    cov_yy = (y_ens-mean(y_ens)) @ (y_ens-mean(y_ens)).T / ens_size

    K = cov_xy@np.linalg.pinv(cov_yy + noise**2*np.eye(*cov_yy.shape))

    if randomized_likelihood and i==0:
      #https://arxiv.org/abs/2507.03207
      y_hat = y + noise * rng.normal(size=y_ens.shape)
    
    x_ens = x_ens + K@(y_hat-y_ens)

    # Just for statistics internally collected in f,
    # evalue model at the mean
    # This also a good output of the model, because
    # Having the last evaluated enseble, we are able
    # to update the parameter vector
    # However, instead of recomputing the full ensemble
    # We just compute it for the mean prediction
    f(mean(x_ens).reshape(-1));

  return mean(x_ens).reshape(-1), dict(x_ens=x_ens_data, y_ens=y_ens_data)
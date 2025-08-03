import numpy as np
def EKI(f, ens_size=None, niters=5, y=0, noise=0, seed=None, uniform=False, low=None, high=None):
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

  if uniform:
    x_ens = np.stack([np.random.uniform(low[i], high[i], ens_size) for i in range(len(low))])
  else:
    rng = np.random.default_rng(seed=seed)
    x_ens = rng.normal(size=(f.dim_of_parameters, ens_size))

  mean=lambda x: x.mean(axis=-1,keepdims=True)
  for i in range(niters):
    y_ens = np.stack([f(x) for x in x_ens.T]).T
    cov_xy = (x_ens-mean(x_ens)) @ (y_ens-mean(y_ens)).T / ens_size
    cov_yy = (y_ens-mean(y_ens)) @ (y_ens-mean(y_ens)).T / ens_size

    K = cov_xy@np.linalg.pinv(cov_yy + noise**2*np.eye(*cov_yy.shape))

    x_ens = x_ens + K@(y-y_ens)

    # Just for statistics internally collected in f,
    # evalue model at the mean
    # This also a good output of the model, because
    # Having the last evaluated enseble, we are able
    # to update the parameter vector
    # However, instead of recomputing the full ensemble
    # We just compute it for the mean prediction
    f(mean(x_ens).reshape(-1));

  return mean(x_ens).reshape(-1)
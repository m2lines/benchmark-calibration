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

import numpy as np

def EKI_Julia(f, ens_size=None, niters=5, y=0, noise=0, seed=None,
              uniform=False, low=None, high=None):
    """
    Calls EnsembleKalmanProcesses.jl with the same interface as EKI().

    Returns:
        mean_est (np.ndarray): mean parameter estimate
        stats (dict): dict with keys 'x_ens' and 'y_ens', 
                      lists of ensembles per iteration
    """
    from julia import Main
    jl = Main
    jl.eval("using EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions")
    jl.eval("using LinearAlgebra, Statistics, Random")

    rng = np.random.default_rng(seed=seed)

    # Observations
    jl.y = y
    jl.eval(f"Γ = {noise**2} * I")

    # Ensemble size
    if ens_size is None:
        ens_size = f.dim_of_parameters * 10

    # Initial ensemble
    if uniform:
      x0 = np.stack([rng.uniform(low[i], high[i], ens_size) for i in range(len(low))])
    else:
      x0 = rng.normal(size=(f.dim_of_parameters, ens_size))
    jl.x0 = x0

    # Define inversion problem
    jl.eval("""
    vanilla_eki = EnsembleKalmanProcess(
        x0, y, Γ, Inversion(),
        scheduler = DefaultScheduler(1),
        accelerator = DefaultAccelerator(),
        localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
        verbose = false
    )
    """)

    # Storage
    x_ens_data = []
    y_ens_data = []

    # Run iterations
    for i in range(niters):
      # pull current ensemble from Julia
      x_ens = np.array(jl.eval("get_u_final(vanilla_eki)"))
      # evaluate forward model in Python, just like in EKI()
      y_ens = np.stack([f(x) for x in x_ens.T]).T

      x_ens_data.append(x_ens)
      y_ens_data.append(y_ens)

      # push results back to Julia for update
      jl.y_ens = y_ens
      jl.eval("update_ensemble!(vanilla_eki, y_ens, deterministic_forward_map=false)")

      f((x_ens).mean(-1).reshape(-1));

    # Final mean estimate
    mean_est = np.array(jl.eval("get_u_mean_final(vanilla_eki)")).reshape(-1)

    stats = dict(x_ens=x_ens_data, y_ens=y_ens_data)
    return mean_est, stats

class Rosenbrock:
    '''
    Vector-valued version of Himmelblau's function:
    f(x,y)=[(a-x), b(y-x^{2})]
    with a=1, b=100
    Has a global minimum:
    (1,1)
    '''
    def __init__(self, noise=0.0, seed=0, a=np.sqrt(7/5), b=1):
        self.noise = noise
        self.calls = 0
        self.rng = np.random.default_rng(seed)
        self.errors = []
        self.x = []
        self.dim_of_parameters = 2
        self.a = a
        self.b = b

    def __call__(self, _x):
        self.calls += 1
        x = np.array(_x)
        self.x.append(x)
        out = np.array([self.a - x[0], self.b * (x[1] - x[0]**2)])
        out += self.noise * self.rng.normal(size=out.shape)
        self.errors.append(np.linalg.norm(x-np.array([self.a,self.a**2])))
        return out

if __name__ == "__main__":
  noise = 0.001
  seed = 42

  # Python EKI
  root_py, stats_py = EKI(Rosenbrock(noise=noise, seed=seed), y=0, noise=noise, seed=seed)

  # Julia EKI
  root_jl, stats_jl = EKI_Julia(Rosenbrock(noise=noise, seed=seed), y=np.zeros(2), noise=noise, seed=seed)

  # Compare results
  print("Python EKI mean:", root_py)
  print("Julia EKI mean:", root_jl)

  diff = np.linalg.norm(root_py - root_jl)
  print("L2 difference between Python and Julia EKI means:", diff)

  if diff < 1e-12:
      print("✅ Python and Julia implementations are equivalent up to numerical precision")
  else:
      print("⚠️ Python and Julia implementations differ")
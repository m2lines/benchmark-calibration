import numpy as np
def EKI(f, ens_size=None, niters=5, y=0, noise=0, seed=None, uniform=False, low=None, high=None, randomized_likelihood=False, x_ens=None,
        julia_backend=False,
        scheduler="DefaultScheduler(1)",
        accelerator="DefaultAccelerator()",
        localization_method="EnsembleKalmanProcesses.Localizers.NoLocalization()",
        deterministic_forward_map="false"):
  '''
  Vanilla Ensemble Kalman Inversion (EKI) implementation is implemented by 
  default in Python and Julia and should be equal up to numerical precision.

  One can try to switch Julia implementation to use additional options:
  scheduler = "DataMisfitController(terminate_at = 1)",
  localization_method = "EnsembleKalmanProcesses.Localizers.SECNice()",
  accelerator = "NesterovAccelerator()"

  deterministic_forward_map="true"
  can be an option but does not work with accelerator
  '''
  if ens_size is None:
    ens_size = f.dim_of_parameters * 10

  rng = np.random.default_rng(seed=seed)
  
  if x_ens is None:
    if uniform:
      x_ens = np.stack([rng.uniform(low[i], high[i], ens_size) for i in range(len(low))])
    else:
      x_ens = rng.normal(size=(f.dim_of_parameters, ens_size))

  mean=lambda x: x.mean(axis=-1,keepdims=True)
  x_ens_data = []
  y_ens_data = []

  if julia_backend:
    from julia import Main
    Main.eval("using EnsembleKalmanProcesses, LinearAlgebra")
    Main.eval(f"Γ = {noise**2} * I")
    Main.x_ens = x_ens
    Main.y = y
    Main.eval(f"""
              eki = EnsembleKalmanProcess(
                  x_ens, y, Γ, Inversion(),
                  scheduler = {scheduler},
                  accelerator = {accelerator},
                  localization_method = {localization_method},
                  verbose = false
              )
              """)

  y_hat = y
  for i in range(niters):
    y_ens = np.stack([f(x) for x in x_ens.T]).T

    x_ens_data.append(x_ens)
    y_ens_data.append(y_ens)

    if julia_backend:
      Main.y_ens = y_ens
      Main.eval(f"update_ensemble!(eki, y_ens, deterministic_forward_map={deterministic_forward_map})")
      x_ens = np.array(Main.eval("get_u_final(eki)"))
    else:
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

class Rosenbrock:
    '''
    Vector-valued version of Himmelblau's function:
    f(x,y)=[(a-x), b(y-x^{2})]
    with a=1, b=100
    Has a global minimum at:
    (a,a**2)
    '''
    def __init__(self, noise=0.0, seed=0, a=np.sqrt(7/5), b=1, A_affine = None, b_affine = None):
        self.noise = noise
        self.calls = 0
        self.rng = np.random.default_rng(seed)
        self.errors = []
        self.x = []
        self.dim_of_parameters = 2
        self.a = a
        self.b = b
        self.A_affine = A_affine
        self.b_affine = b_affine

    def __call__(self, _x):
        self.calls += 1
        x = np.array(_x)
        if self.A_affine is not None and self.b_affine is not None:
           x = np.linalg.inv(self.A_affine) @ (x - self.b_affine)
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
  root_jl, stats_jl = EKI(Rosenbrock(noise=noise, seed=seed), y=np.zeros(2), noise=noise, seed=seed, julia_backend=True)

  # Compare results
  print("Python EKI mean:", root_py)
  print("Julia EKI mean:", root_jl)

  diff = np.linalg.norm(root_py - root_jl)
  print("L2 difference between Python and Julia EKI means:", diff)

  if diff < 1e-12:
      print("✅ Python and Julia implementations are equivalent up to numerical precision")
  else:
      print("⚠️ Python and Julia implementations differ")
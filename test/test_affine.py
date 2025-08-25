import sys
import numpy as np
sys.path.append('../src')

from ensemble_kalman_inversion import EKI, Rosenbrock

noise = 0.001
seed = 42

x_ens = np.random.randn(2,20)

# Untransformed iteration
root, stats = EKI(Rosenbrock(noise=noise, seed=seed), y=np.zeros(2), noise=noise, seed=seed, x_ens = x_ens,
                  julia_backend=True,
                  #scheduler = "DataMisfitController(terminate_at = 1)",
                  localization_method = "EnsembleKalmanProcesses.Localizers.SECNice()",
                  #accelerator = "NesterovAccelerator()"
                  )

# Introduce Affine transformation
A = np.random.randn(2,2)
b  = np.random.randn(2)

x_ens_transformed = A @ x_ens + b[:,None]

root_t, stats_t = EKI(Rosenbrock(noise=noise, seed=seed, A_affine=A, b_affine=b), y=np.zeros(2), noise=noise, seed=seed, x_ens=x_ens_transformed,
                      julia_backend=True,
                      #scheduler = "DataMisfitController(terminate_at = 1)",
                      localization_method = "EnsembleKalmanProcesses.Localizers.SECNice()",
                      #accelerator = "NesterovAccelerator()"
                      )

root_transformed = np.linalg.inv(A) @ (root_t - b)

# Compare results
print("Untransformed EKI mean:", root)
print("Transformed EKI mean:", root_transformed)

diff = np.linalg.norm(root - root_transformed)
print("L2 difference between Transformed and Untransformed EKI means:", diff)

if diff < 1e-12:
    print("✅ Affine transformed is equivalent to untransformed up to numerical precision")
else:
    print("⚠️ Affine transformed is different from untransformed")
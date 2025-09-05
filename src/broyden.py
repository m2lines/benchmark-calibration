import numpy as np

def broyden_solver(f, x0, niters=20, lambda_regularization=0, bad_broyden=False, minimum_step=1e-15, verbose=False, chord=False, fixed=False, gradient_descend_step=0.0, gradient_descend_with_newton_step=False):
    """
    Broyden's method for solving f(x) = 0
    Parameters:
      f: function from R^m to R^n with n>=m
      x0: initial guess (1D np.array)
      niters: maximum number of iterations
      noise: standard deviation of noise in forward model
    Returns:
      x: estimated solution
    """
    x = x0.copy()

    m = len(x)
    y = f(x)
    y0 = y
    n = len(y)

    # Simple initialization with finite differences and relatively significant
    # step. Ideally, taken from the prior distribution

    J = np.zeros((n,m))
    H = np.zeros((m,n))
    skipped_steps = 0
    for k in range(m):
      h = np.zeros(m)
      h[k] = 1.0  # unit step in k-th parameter
      x_new = x + h
      y_new = f(x_new)
      w = y_new - y
      J += np.outer(w-J@h, h) / np.dot(h,h)

    H = np.linalg.pinv(J)

    for i in range(niters):
        if gradient_descend_step > 0.0:
          h = - gradient_descend_step * J.T@y
        else:
          if bad_broyden:
            h = - H@y
          else:
            h = - np.linalg.solve(J.T@J + lambda_regularization * np.eye(m,m), J.T@y)

        if (np.linalg.norm(h) / np.sqrt(m) < minimum_step):
          print('Stop. Update is too small at iteration', i)
          break

        x_new = x + h
        y_new = f(x_new)

        if not(fixed):
          if chord:
            w = y_new - y0
            h = x_new - x0
          else:
            w = y_new - y
            h = x_new - x

          if bad_broyden:
            H += np.outer(h-H@w, w) / np.dot(w,w)
          else:
            J += np.outer(w-J@h, h) / np.dot(h,h)

        x, y = x_new, y_new
    return x

def Gauss_Newton(f, x0, niters=20):
  """
  This is Gauss-Newton solver which assumes
  that the Jacobian is known and deterministic but
  the residual can be stochastic
  """
  x = x0.copy()

  for i in range(niters):
    y = f(x)
    J = f.J(x)
    x = x - np.linalg.solve(J.T@J, J.T@y)
    
  return x
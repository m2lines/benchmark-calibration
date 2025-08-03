import numpy as np

def broyden_solver(f, x0, niters=20, lambda_regularization=0, bad_broyden=False, minimum_step=1e-15, verbose=False):
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

    loss_ratio_expected = []
    loss_ratio_final = []

    for i in range(niters):
        if bad_broyden:
          h = - H@y
          Jh = np.linalg.pinv(H)@h
        else:
          h = - np.linalg.solve(J.T@J + lambda_regularization * np.eye(m,m), J.T@y)
          Jh = J@h

        if (np.linalg.norm(h) / np.sqrt(m) < minimum_step):
          print('Stop. Update is too small at iteration', i)
          break

        initial_loss = np.linalg.norm(y)**2
        expected_loss = np.linalg.norm(y+Jh)**2

        x_new = x + h
        y_new = f(x_new)

        final_loss = np.linalg.norm(y_new)**2
        loss_ratio_expected.append(expected_loss/initial_loss)
        loss_ratio_final.append(final_loss/initial_loss)
        if verbose:
          print('LOSS RATIO>Expected/Final: %.2f/%.2f' % (expected_loss/initial_loss, final_loss/initial_loss))
          try:
            Jh_true = f.J@h
            print('Jh error', np.linalg.norm(Jh_true-Jh)/np.linalg.norm(Jh_true))
          except:
            pass

        w = y_new - y

        if bad_broyden:
          H += np.outer(h-H@w, w) / np.dot(w,w)
        else:
          J += np.outer(w-J@h, h) / np.dot(h,h)

        x, y = x_new, y_new
    return x
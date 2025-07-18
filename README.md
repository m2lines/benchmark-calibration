The purpose of this repository is to benchmark various calibration methods designed to solve inverse problems on simple analytical examples.

# Inverse problem
The inverse problem has the following building blocks:
* $y=G(u)$ is the forward map
* $u\in{R}^{N_u}$ is the vector of parameters to be calibrated
* $y\in{R}^{N_y}$ is a vector of observations
* Generally, $N_u \neq N_y$
* Forward map is often noisy, meaning that $G(u)$ evaluated twice is expected to give different answers
* $\Gamma$ is the covariance matrix of the forward map noise

Note: the forward map is assumed to be not differentiable but can be evaluated several times O(100-1000) in order to solve the Inverse Problem defined below.

## Definition of inverse problem
Given vector $y$, map $G$ and, optionally, covariance matrix $\Gamma$, find vector of parameters $u$ which fits equation $y=G(u)$ best.

## List of deterministic inverse problems ($\Gamma=0$)
Inverse problems, even linear ones, are famous for having many caveats related to the non-existence or non-uniqueness of the solution. Below, we suggest a set of simple analytical inverse problems, to be used for benchmarking of calibration methods.

Find the minimum of a quadratic function:
* $G(u)=\sum_{i=1}^n(u_i-a_i)^2$, $u, a\in{R}^n$
* $y=0$
* **Answer**: $u=a$.
  
Find a unit sphere (non-unique solution):
* $G(u)=\sum_{i=1}^n(u_i-a_i)^2$, $u, a\in{R}^n$
* $y=1$
* **Answer**: $u$ in unit sphere with center at $u=a$.

Transcendental equation:
* $G(u) = e^{-u}-u$, $u\in{R}^1$
* $y=0$
* **Answer**: $u=0.56714329$

Coordinatewise transcendental equation:
* $G(u) = e^{-u}-u$, $u\in{R}^n$
* $y=0 \in {R}^n$
* **Answer**: $u=0.56714329[1,1,\cdots]^T$

One-coordinate transcendental equation:
* $G(u) = e^{-u_k}-u_k$, $u\in{R}^n$
* $y=0$
* **Answer**: All $u$ with $u_k=0.56714329$.

Non-coordinatewise transcendental equation:
* $G(u) = e^{-Au}-Au$, $u\in{R}^n, A \in R^{n \times n}$
* $y=0 \in {R}^n$
* **Answer**: $u=0.56714329A^{-1}[1,1,\cdots]^T$

Identity linear problem:
* $G(u) = u$, $u \in {R}^n$
* $y=[\underbrace{0,\cdots, 1}_k,\cdots,0]^T\equiv e_k$
* **Answer**: $u=e_k$

Underdetermined rank-one linear system:
* $G(u) = e_k^T u$, where $u, e_k \in {R}^n$
* $y=1$
* **Answer**: All $u$ with $u_k=1$

Overdetermined rank-one linear system - unique solution:
* $G(u) = e_k u$, where $u \in R^{1}$ and $e_k \in {R}^n$
* $y=e_k$
* **Answer**: $u=1$

Overdetermined rank-one linear system - optimal solution:
* $G(u) = e_k u$, where $u \in R^{1}$ and $e_k \in {R}^n$
* $y=e_k + e_K$, where $K \neq k$
* **Answer**: $u=1$ is the optimal solution (in MSE); an exact solution does not exist


# Calibration methods



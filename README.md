The purpose of this repository is to benchmark various calibration methods on simple inverse problems.

# Inverse problem
We must preliminarily define the following building blocks of the inverse problem:
* $y=G(u)$ is the forward map
* $u\in\mathbb{R}^n$ is the vector of parameters to be calibrated
* $y\in\mathbb{R}^k$ is a vector of observations
* Forward map is often noisy, meaning that $G(u)$ evaluated twice is expected to give different answers
* $\Gamma$ is the covariance matrix of the forward map noise

Note: the forward map is assumed to be not differentiable but can be evaluated several times O(100-1000) in order to solve the Inverse Problem given below.

## Definition of inverse problem
Given vector $y$, map $G$ and, optionally, covariance matrix $\Gamma$, find vector of parameters $u$ which fits equation $y=G(u)$ best.


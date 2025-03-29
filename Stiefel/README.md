## What is the Stiefel manifold?

The [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold), denoted by $St(n,m)$ is the set of all $n\times m$ matrices with orthonormal columns, i.e.,
$$St(n, m):=\left\{X\in\mathbb{R}^{n\times m}: X^\top X=I\right\}$$

This folder contains code for optimizing on the Stiefel manifold. [VariationalStiefelSGD](./VariationalStiefelSGD.m)  implements several gradient descent with momentum algorithms,  including versions analogous to NAG method and heavy-ball method in Euclidean space. [VariationalStiefelAdam](./VariationalStiefelAdam.m) contains Adam algorithm, an adaptive learning rate algorithm on the Stiefel manifold. Among them, NAG has the optimal convergence rate, heavy-ball is suitable for problems with stochastic gradient and Adam is suitable for large scale machine learning tasks.

**NAG is our newly developed algorithm and empirically has the optimal convergence rate. Comparing to heavy-ball in this [paper](https://arxiv.org/pdf/2205.14173), it performs better on problems with large condition numbers.**


## Why our algorithms?

There are many algorithms designed for general manifolds, and also other approaches, e.g., penalty. Our algorithm for Lie groups are good due to the following reasons:
- Preserve the geometric structure strictly. 
- The optimizer has optimal convergence rate under gradient oracle. 
- Empirically, it is suitable for a wide range of tasks, including stochastic gradient, large-scale neural network and scientific computing.


## Usage

```
% get a n-by-m random Stiefel matrix
X_init=InitStiefelMatrix(n,m);

[loss, grad]=@(x)f_grad(x)
    loss=...
    grad=...
end

hp={};
% adjust hyperparameter here.

% for optimization
[X, out]=VariationalStiefelSGD(X_init, @f_grad, hp);
```

where `X_init` is the initial value for optimization and `f_grad` is a function with input an n-by-m matrix `x` and output loss and gradient.

For detailed examples, please see [leading eigenvalue example](./example_LEV_stiefel.m).


### Details

Given a Stiefel manifold $St(n,m)$ and an objective function $f: St(n,m)\to \mathbb{R}$, an optimization problem is

$$\min_{x\in St(n,m)} f(x)$$

Example: Leading EigenValue decomposition (LEV) problem: Given a $n\times n$ symmetric matrix $A$, we want to calculate its $m$ largest eigenvalue. Such problem can be formulated to an optimization on $St(n,m)$ as following:

$$\min_{X\in St(n, m)} \operatorname{tr} X^\top A X N, \quad N=\operatorname{diag}(1, \dots, m)$$

Usage of `VariationalStiefelSGD` (`VariationalStiefelAdam` is similar): `[g, out]=VariationalStiefelSGD(X_init, @f_grad, hp)` where the input `X_init` is a n-by-m matrix, `f_grad` is the objective function and `hp` is the hyperparameter. The output `X` is the minimum and `out` is a dict containing the training curve. Please see the following for the choice of hyperparameter:

`hp` in `VariationalLieSGD` has the following attributes:
- `h`: step size. Positive float number. Default: 0.1
- `gamma`: energy dissipation. Positive float number. Default: 0.01
- `algo`: specify the algorithm to be used. Should be a string in 'nag_c', 'nag_sc', 'heavy_ball' 'momentum_free'. Default: 'nag_c'
- `a`: a float number with $a<1$ define the norm in the tangent bundle of the Stiefel manifold. Please see Eq. 2 in [this paper](https://arxiv.org/pdf/2205.14173). Usually, the default choice is good enough. Default: 0.5
- `expm_method`: specify how to approximate the matrix exponential. Should be a string in 'ForwardEuler', 'Cayley', 'MatrixExp'. Usually, this has no significant influence on the performance and 'ForwardEuler' is the cheapest in computation. Default: 'ForwardEuler'
- `max_iter`: the algorithm will terminate when this number of iteration is reached. Positive integer. Default: 1000
- `gtol`: the algorithm will terminate when the gradient is less than this value. non-negative float number. Default: 0.01
- `restart`: whether to use the restart scheme. Default: true. The restart scheme we use is in Sec 5 in [this paper](https://www.jmlr.org/papers/volume17/15-084/15-084.pdf).
- `verbose`: whether to enable printing. Default: false


### Our recommendation for hyperparameter:

- For machine learning tasks, e.g., train neural networks,  we recommend a default setting of algo=nag_sc/heavy_ball and restart=false.
- For tasks with stochastic/noisy gradient,  we recommend a default setting of algo=nag_sc and restart=false.
- Otherwise, algo=nag_c and restart=false will be a good default setting.
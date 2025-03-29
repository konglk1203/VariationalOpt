## What is a Lie group?

A [Lie group](https://en.wikipedia.org/wiki/Lie_group) is a manifold with group structure. This folder contains code for optimizing and sampling on Lie groups. An example for Lie group:
$$SO(n):=\left\{X\in\mathbb{R}^{n\times n}: X^\top X=I\right\}$$


## Why our algorithms?

There are many algorithms designed for general manifolds, and also other approaches, e.g., penalty. Our algorithm for Lie groups are good due to the following reasons:
- Preserve the geometric structure strictly. 
- The KLMC sampler has mathematically proved convergence guarantee. This is the only kinetic Langevin sampler on curved space with proved convergence.
- The optimizer has quantified convergence rate, which is optimal under gradient oracle. 
- Empirically, it is suitable for a wide range of tasks, including stochastic gradient, large-scale neural network and scientific computing.


## Usage

```
G=YourLieGroup(args)

[loss, grad]=@(x)f_grad(x)
    loss=...
    grad=...
end

hp={};
% adjust hyperparameter here.

% for optimization
[g, out]=VariationalLieSGD(G, @f_grad, hp);

% for sampling
[g, out]=VariationalLieKLMC(G, @f_grad, hp);
```

where `G` is a `LieGroup` object and `f_grad` is a function with input `x` as an element on `G` and output loss and gradient.

For detailed examples, please see [optimization example](./example_EV_Lie_group.m) and [sampling example](./example_KLMC_Lie_group.m).


### Details: defining the Lie group

There are several operations that we need to define for construct a Lie group, e.g., group multiplication. So please inherit your own Lie group from abstract class `LieGroup`. An example for constructing a Lie group is `SOn`.


### Optimization

Given a Lie group $G$ and an objective function $f: G\to \mathbb{R}$, an optimization problem on a Lie group is

$$\min_{g\in G} f(g)$$

Example: EigenValue decomposition (EV) problem: Given a $n\times n$ symmetric matrix $A$, calculating its eigenvalue can be formulated by 
$$\min_{X\in SO(n)} \operatorname{tr} X^\top A X N, \quad N=\operatorname{diag}(1, \dots, n)$$

Usage of `VariationalLieSGD`: `[g, out]=VariationalLieSGD(G, @f_grad, hp)` where the input `G` is a `LieGroup` class, `f_grad` is the objective function and `hp` is the hyperparameter. The output `g` is the minimum and `out` is a dict containing the training curve. Please see the following for the choice of hyperparameter:

`hp` in `VariationalLieSGD` has the following attributes:
- `h`: step size. Positive float number. Default: 0.1
- `gamma`: energy dissipation. Positive float number. Default: 0.01
- `g_init`: initial starting point. Should be a value in the Lie group `G`. Default: group identity.
- `algo`: specify the algorithm to be used. Should be a string in 'nag_c', 'nag_sc', 'heavy_ball' 'momentum_free'. Default: 'nag_c'
- `max_iter`: the algorithm will terminate when this number of iteration is reached. Positive integer. Default: 1000
- `gtol`: the algorithm will terminate when the gradient is less than this value. non-negative float number. Default: 0.01
- `restart`: whether to use the restart scheme. Default: true. The restart scheme we use is in Sec 5 in [this paper](https://www.jmlr.org/papers/volume17/15-084/15-084.pdf)
- `verbose`: whether to enable printing. Default: false


### Our recommendation for hyperparameter:

- For machine learning tasks, e.g., train neural networks,  we recommend a default setting of algo=nag_sc/heavy_ball and restart=false.
- For tasks with stochastic/noisy gradient,  we recommend a default setting of algo=nag_sc and restart=false.
- Otherwise, algo=nag_c and restart=false will be a good default setting.


### Sampling

Given a Lie group $G$ and an potential function $f: G\to \mathbb{R}$, a sampling problem on a Lie group is

$$\text{sample } x\sim \mu, \quad \text{where }\mu(g)\propto\exp(-f(g))$$

`hp` in `VariationalLieKLMC` has the following attributes:
- `h`: step size. Positive float number. Default: 0.1
- `gamma`: energy dissipation. Positive float number. Default: 0.01
- `g_init`: initial starting point. Should be a value in the Lie group `G`. Default: group identity.
- `max_iter`: the algorithm will terminate when this number of iteration is reached. Positive integer. Default: 1000
- `warmup_steps`: the first number of steps will be removed. Default: true
- `mh_correction`: whether to use Metropolis–Hastings correction. Default: 100
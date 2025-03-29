VariationalOpt is a Matlab toolbox for optimization and sampling on curved spaces. Currently, we have optimization and sampling algorithm on Lie groups and also optimization algorithms on the Stiefel manifold (i.e., optimization under orthogonal constraints).


## Why our algorithms?
How to add momentum to Riemannian optimization and sampling algorithms for accelerated convergence is a research frontier. Comparing to existing Matlab packages, this package implements effective momentum versions of optimizers and samplers that were published in the recent couple of years. The algorithms have the following benefits:
- All the algorithms strictly preserve the geometric structure.
- The optimizers has optimal convergence rate under gradient oracle. 
- Empirically, it is suitable for a wide range of tasks, including stochastic gradient, large-scale neural network and scientific computing.
- The sampler implemented has rigorously proved convergence guarantee.

GitHub: [VariationalOpt](https://github.com/konglk1203/VariationalOpt)

Author: Lingkai Kong (Email: lkong75@gatech.edu)


## TL;DR

- Optimization tasks on the Stiefel manifold: For machine learning tasks,  we recommend a default setting of `VariationalStiefelSGD` with `hp.algo='nag_sc'` and `hp.restart=false`. For all other tasks,  we recommend a default setting of `VariationalStiefelSGD` with default hyperparameter.

- Optimization tasks on Lie groups: Please define a Lie group as a subclass of the abstract class `LieGroup` and use `VariationalLieSGD` with default hyperparameter.

- Sampling tasks on Lie groups: Please define a Lie group as a subclass of the abstract class `LieGroup` and use `VariationalLieSGD` with default hyperparameter.


## Quick start guide

Download `VariationalOpt` directory to `/my/directory/VariationalOpt/`. Then go to `/my/directory/VariationalOpt/` and run `import`.

The code are separated to 2 parts: the folder `Lie_group` contains Lie group version of 
- Gradient descent
- Gradient descent with momentum
- Nesterov's accelerate gradient (NAG, a gradient-based optimization algorithm with proved optimal convergence rate)
- Kinetic Langevin Monte Carlo (KLMC, a gradient-based sampling algorithm with convergence guarantee)

The folder `Stiefel` contains optimization algorithms on Stiefel manifold.
- Gradient descent
- Gradient descent with momentum
- Nesterov's accelerate gradient (NAG, a gradient-based optimization algorithm with optimal convergence rate)
- Adam (an adaptive learning rate algorithm that is suitable for machine learning task)


## Examples

`Lie_group/example_LEV_Lie_group` is an example for Lie group optimization. The task is computing the full eigenvalue decomposition of a $n\times n$ symmetric matrix by viewing it as a optimization problem on $SO(n)$.

`Stiefel/example_LEV_stiefel` is an example for Stiefel optimization. Similar to the Lie group, the task is computing the largest $m$ eigenvalues of a $n\times n$ symmetric matrix by viewing it as a optimization problem on the Stiefel manifold $St(n, m)$.

`Lie_group/example_KLMC_Lie_group` contains examples for sampling from $SO(n)$. In this example, we construct a toy potential function $f(x)=-10 x(1,1)^2$ for $x\in SO(10)$ and sample the corresponding Gibbs distribution $\mu\propto \exp(-f)$.


## Folder structure
```
VariationalOpt
│
├── Lie_group                       # Lie group optimization and sampling algorithms
│   ├── VariationalLieSGD.m         # Gradient descent with momentum algorithms on Lie groups
│   ├── VariationalLieKLMC.m        # Kinetic Langevin Monte Carlo sampler on Lie groups
│   ├── LieGroup.m                  # Base class for Lie group. A LieGroup object is required to use the algorithms.
│   ├── MatrixLieGroup.m            # Implement matrix Lie groups, an example for subclass of LieGroup.
│   ├── SOn.m                       # Implement SO(n), servers as an example.
│   ├── example_EV_Lie_group.m      # Example: solving eigenvalue decomposition problem by VariationalLieSGD
│   └── example_KLMC_Lie_group.m    # Example: sampling a toy example on SO(n) by VariationalLieKLMC
│
└── Stiefel                         # Stiefel optimization algorithms
    ├── VariationalStiefelSGD.m     # Gradient descent with momentum algorithms on Stiefel manifold
    ├── VariationalStiefelAdam.m    # adaptive learning rate algorithm on Stiefel manifold 
    ├── example_LEV_stiefel.m       # Example: solving leading eigenvalue decomposition problem by Stiefel optimizers 
    ├── InitStiefelMatrix.m         # initialize a Stiefel matrix randomly 
    └── cayley.m                    # Implement Cayley map, an approximation of matrix exponential
    
    
```
## Citation

The algorithm implemented in this repository are from the following papers.
```bibtex
@article{tao2020variational,
  title={Variational optimization on {L}ie groups, with examples of leading (generalized) eigenvalue problems},
  author={Tao, Molei and Ohsawa, Tomoki},
  journal={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2020}
}

@article{kong2023momentum,
  title={Momentum {S}tiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport},
  author={Kong, Lingkai and Wang, Yuqing and Tao, Molei},
  journal={The International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{kong2024convergence,
  title={Convergence of kinetic {L}angevin {M}onte {C}arlo on {L}ie groups},
  author={Kong, Lingkai and Tao, Molei},
  journal={The Annual Conference on Learning Theory (COLT)},
  year={2024}
}

@article{kong2024quantitative,
  title={Quantitative Convergences of {L}ie Group Momentum Optimizers},
  author={Kong, Lingkai and Tao, Molei},
  journal={The Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
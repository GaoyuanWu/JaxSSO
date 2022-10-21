# JaxSSO
A framework for structural shape optimization based on automatic differentiation (AD) and the adjoint method, enabled by [JAX](https://github.com/google/jax).

## Features
* Automatic differentiation (AD): an easy and accurate way for gradient evaluation. The implementation of AD avoids deriving derivatives manually or trauncation errors from numerical differentiation.
* Acclerated linear algebra (XLA) and just-in-time compilation: these features in JAX boost the gradient evaluation
* Hardware acceleration: run on GPUs and TPUs for **faster** experience.

## Background: shape optimization
We consider the minimization of the ***strain energy*** by changing the **shape** of structures, which is equivalent to maximizing the stiffness and reducing the
bending in the structure. The mathematical formulation of this problem is as follows, where no additional constraints are considered.
$$\text{minimize} \quad C(\mathbf{x}) = \frac{1}{2}\int\sigma\epsilon \mathrm{d}V = \frac{1}{2}\mathbf{f}^\mathrm{T}\mathbf{u}(\mathbf{x}) $$
$$\text{subject to: } \quad \mathbf{K}(\mathbf{x})\mathbf{u}(\mathbf{x}) =\mathbf{f}$$
where $C$ is the compliance, which is equal to the work done by the external load; $\mathbf{x} \in \mathbb{R}^{n_d}$ is a vector of $n_d$ design variables that determine the shape of the structure; $\sigma$, $\epsilon$ and $V$ are the stress, strain and volume, respectively; $\mathbf{f} \in \mathbb{R}^n$ and $\mathbf{u}(\mathbf{x}) \in \mathbb{R}^n$ are the generalized load vector and nodal displacement of $n$ structural nodes; $\mathbf{K} \in \mathbb{R}^{6n\times6n}$ is the stiffness matrix. The constraint is essentially the governing equation in finite element analysis (FEA).

To implement **gradient-based optimization**, one needs to calculate $\nabla C$. By applying the ***adjoint method***, the entry of $\nabla C$ is as follows:
$$\frac{\partial C}{\partial x_i}=-\frac{1}{2}\mathbf{u}^\mathrm{T}\frac{\partial \mathbf{K}}{\partial x_i}\mathbf{u}$$ The use of the adjoint method: i) reduces the computation complexity and ii) decouples FEA and the derivative calculation of the stiffness matrix $\mathbf K$.
To get $\nabla C$:
1. Conduct FEA to get $\mathbf u$
2. Conduct sensitivity analysis to get $\frac{\partial \mathbf{K}}{\partial x_i}$. which is done by the module in JaxSSO called `Model_Sens.py` using AD.


# JaxSSO
A framework for structural shape optimization based on automatic differentiation (AD) and the adjoint method, enabled by [JAX](https://github.com/google/jax).

Developed by [Gaoyuan Wu](https://gaoyuanwu.github.io/) @ Princeton.


We have a [preprint](https://doi.org/10.48550/arXiv.2211.15409) under review where you can find details regarding this framework.
Please share our project with others and cite us if you find it interesting and helpful.
Cite us using:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.15409,
  doi = {10.48550/ARXIV.2211.15409},
  url = {https://arxiv.org/abs/2211.15409},
  author = {Wu, Gaoyuan},
  title = {A framework for structural shape optimization based on automatic differentiation, the adjoint method and accelerated linear algebra},
  publisher = {arXiv},
  year = {2022},
}

```
## Features
* Automatic differentiation (AD): an easy and accurate way for gradient evaluation. The implementation of AD avoids deriving derivatives manually or trauncation errors from numerical differentiation.
* Acclerated linear algebra (XLA) and just-in-time compilation: these features in JAX boost the gradient evaluation
* Hardware acceleration: run on GPUs and TPUs for **faster** experience.
* Form finding based on finite element analysis (FEA) and optimization theory

Here is an implementation of JaxSSO to form-find a structure inspired by [Mannheim Multihalle](https://mannheim-multihalle.de/en/architecture/) using simple gradient descent. (First photo credit to Daniel Lukac)
![alt text](https://github.com/GaoyuanWu/JaxSSO/blob/main/data/images/MannheimMultihalle.jpg)
![alt text](https://github.com/GaoyuanWu/JaxSSO/blob/main/data/images/MM_opt.jpg)
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

## Usage

### Installation
Install it with pip: `pip install JaxSSO`

### Dependencies
JaxSSO is written in Python and requires:
* [numpy](https://numpy.org/doc/stable/index.html)
* [JAX](https://jax.readthedocs.io/en/latest/index.html): "JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance machine learning research." Please refer to [this link](https://github.com/google/jax#installation) for the installation of JAX.
* [Nlopt](https://nlopt.readthedocs.io/en/latest/): Nlopt is a library for nonlinear optimization. It has Python interface, which is implemented herein. Refer to [this link](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/) for the installation of Nlopt. Alternatively, you can use `pip install nlopt`, please refer to [
nlopt-python](https://pypi.org/project/nlopt/).

Optional:
* [PyNite](https://github.com/JWock82/PyNite): A 3D structural engineering finite element library for Python, which is used herein for structural analysis. You choose any FEA solver you'd prefer (e.g. opensees) and couple it with JaxSSO.

### Quickstart
The project provides you with interactive examples with Google Colab for quick start:
* [2D-arch](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Arch_2D.ipynb): form-finding of a 2d-arch
* [3D-arch](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Arch_3D.ipynb): form-finding of a 3d-arch
* [Mannheim Multihalle](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Mannheim_Multihalle.ipynb): form-finding of Mannheim Multihalle
* [Four-point supported gridshell](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/FourPt_FreeForm.ipynb): form-finding of a gridshell with four coner nodes pinned. The geometry is parameterized by Bezier Surface.
* [Two-edge supported canopy, unconstrained](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/TwoEdge_FreeForm_Unconstrained.ipynb): form-finding of a canopy. The geometry is parameterized by Bezier Surface.
* [Two-edge supported canopy, constrained](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/TwoEdge_FreeForm_Constrained.ipynb): form-finding of a canopy with height constraints. The geometry is parameterized by Bezier Surface.

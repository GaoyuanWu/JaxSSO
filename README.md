# JaxSSO
A framework for structural shape optimization based on automatic differentiation (AD) and the adjoint method, enabled by [JAX](https://github.com/google/jax).

Developed by [Gaoyuan Wu](https://gaoyuanwu.github.io/) @ Princeton.


We have a [recent publication](https://link.springer.com/article/10.1007/s00158-023-03601-0) in [Structural and Multidisciplinary Optimization](https://www.springer.com/journal/158) where you can find details regarding this framework.
Please share our project with others and cite us if you find it interesting and helpful.
Cite us using:
```bibtex
@article{wu_framework_2023,
	title = {A framework for structural shape optimization based on automatic differentiation, the adjoint method and accelerated linear algebra},
	volume = {66},
	issn = {1615-1488},
	url = {https://doi.org/10.1007/s00158-023-03601-0},
	doi = {10.1007/s00158-023-03601-0},
	language = {en},
	number = {7},
	urldate = {2023-06-21},
	journal = {Structural and Multidisciplinary Optimization},
	author = {Wu, Gaoyuan},
	month = jun,
	year = {2023},
	keywords = {Adjoint method, Automatic differentiation, Bézier surface, Form finding, JAX, Shape optimization, Shell structure},
	pages = {151},
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
2. Conduct sensitivity analysis to get $\frac{\partial \mathbf{K}}{\partial x_i}$.

## Usage

### Installation
Install it with pip: `pip install JaxSSO`

### Dependencies
JaxSSO is written in Python and requires:
* [numpy](https://numpy.org/doc/stable/index.html) >= 1.22.0.
* [JAX](https://jax.readthedocs.io/en/latest/index.html): "JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance machine learning research." Please refer to [this link](https://github.com/google/jax#installation) for the installation of JAX.
* [Nlopt](https://nlopt.readthedocs.io/en/latest/): Nlopt is a library for nonlinear optimization. It has Python interface, which is implemented herein. Refer to [this link](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/) for the installation of Nlopt. Alternatively, you can use `pip install nlopt`, please refer to [
nlopt-python](https://pypi.org/project/nlopt/).
* [scipy](https://scipy.org/).


### Quickstart
The project provides you with interactive examples with Google Colab for quick start:
* [2D-arch](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Arch_2D.ipynb): form-finding of a 2d-arch
* [3D-arch](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Arch_3D.ipynb): form-finding of a 3d-arch
* [Mannheim Multihalle](https://github.com/GaoyuanWu/JaxSSO/blob/main/Examples/Mannheim_Multihalle.ipynb): form-finding of Mannheim Multihalle

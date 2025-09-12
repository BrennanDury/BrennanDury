# Hi, I'm Brennan 👋

[![Email](https://img.shields.io/badge/Email-Contact-informational?logo=gmail)](mailto:brennandury@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/brennan-dury-8a8804224/)

---

## About Me 🧑‍💻

I'm a master's student in mathematics at the University of Pennsylvania and a machine learning researcher. In the projects below, I develop, evaluate, and optimize transformer designs for physical trajectory prediction and Bayesian models for cell-state differentiation and genomics.

---

## Projects 🚀

### [Transformer Package for Physics](https://github.com/eg-trim/trim-transformer)
I built a PyPI package that replicates the interface of `torch.nn.TransformerEncoder` but with a custom attention layer and support for key-value caching. I also designed a novel [algorithm](https://github.com/eg-trim/trim-transformer/blob/triton/trim_transformer/functional.py) to compute each attention head with time $\Theta(nd^2)$ and memory $\Theta(nd + d^2)$, where $n$ is the sequence length and $d$ is the head dimension. With an attention mask, previous algorithms were $\Theta(nd^2)$ in memory. There is no way to compute this algorithm using Pytorch operations without for loops, so I wrote a custom kernel using Triton.

---

### [Bayesian Modeling for Genomics](https://github.com/settylab/2for1separator)
For a Gaussian process, we needed to perform linear algebra operations on a large covariance matrix. Because genomic data is long relative to the scale at which interactions are relevant in our case, the covariance matrix has mostly near-zero entries. I designed an [algorithm](https://github.com/settylab/2for1separator/blob/main/sep241/sep241covariance.py) to efficiently compute a sparse approximation to the covariance matrix. The sparse approximation is not guaranteed to be positive definite, so I designed and proved a lower bound on the eigenvalues using diagonal dominance and added a scalar multiple of the identity matrix to guarantee positive definiteness, see [auto_jitter](https://github.com/settylab/2for1separator/blob/main/sep241/sep241covariance.py). I also refactored this repository to build a PyPI package and documentation.

---

### [Bayesian Modeling for Cell States](https://github.com/settylab/Mellon)
I developed various optimization tricks for a Bayesian model of cell-state differentiation. This model also featured a large Gaussian process, but not one dimensional. Based on a literature review and experimentation, I found that a [modified](https://arxiv.org/abs/1708.03218) Nystrom method with k-means [landmarks](https://proceedings.mlr.press/v70/oglic17a.html) was an effective solution to reduce the complexity from $\Theta(n^3)$ to $\Theta(nk^2)$, where $n$ is the number of samples and $k$ is the number of landmarks, without noticably affecting performance. See my [implementation](https://github.com/settylab/Mellon/blob/effcfb4ba354190a79407f0a0b48167300783fd5/Crowding/decomposition.py). Alongside switching to Jax from PyMC, discovering that Maximum A Posteriori optimization was approximately as accurate as full MCMC sampling, and designing an [initial guess](https://github.com/settylab/Mellon/blob/effcfb4ba354190a79407f0a0b48167300783fd5/Crowding/parameters.py) for optimization, these optimizations enabled us to run experiments faster and run larger experiments, which enabled us to develop data-driven [heuristics](https://github.com/settylab/Mellon/blob/effcfb4ba354190a79407f0a0b48167300783fd5/Crowding/parameters.py) for automatic selection of hyperparameters, saving further time by reducing or eliminating the hyperparameter search. I also built this repository as a PyPI package with documentation. See the code at this [commit](https://github.com/settylab/Mellon/tree/effcfb4ba354190a79407f0a0b48167300783fd5) for my work. 

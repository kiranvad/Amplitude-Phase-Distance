# Amplitude-Phase-Distance
A light-weight repository to compute Amplitude Phase distance between two functions

## Installation

1. Installation of this package is done in two steps. First, install the package using the following:

```bash
pip install git+https://github.com/kiranvad/Amplitude-Phase-Distance.git
```

2. Now, install a Cython-based package called warping:

```bash
pip install git+https://github.com/kiranvad/warping.git
```

3. If you would like to use this function in PyTorch, install the following:
```bash
pip install git+https://github.com/kiranvad/funcshape.git
```

You should now be able to run the [example](/example.ipynb)


## Notes
This package is intended to be used as a faster alternative to original works in [fdasrsf-python](https://github.com/jdtuck/fdasrsf_python).
If you used this package, please cite the original `fdasrsf-python` package using the following:

```bib
@phdthesis{tucker2014functional,
  title={Functional component analysis and regression using elastic methods},
  author={Tucker, J Derek},
  year={2014},
  school={The Florida State University}
}
```

For applications in materials science-related problems, you can look at the following papers or repositories linked there:

1. Vaddi, Kiran, Huat Thart Chiang, and Lilo D. Pozzo. "Autonomous retrosynthesis of gold nanoparticles via spectral shape matching." Digital Discovery (2022).

Introduces amplitude-phase distance to engineering audience and apply it to create a self-driving lab synthesizing gold nanoparticles matching the shape of target UV-Vis spectra.

2. Lachowski, Kacper J., et al. "Multivariate analysis of peptide-driven nucleation and growth of Au nanoparticles." Digital Discovery (2022).

Introduces functional principle component analysis to UV-Vis spectroscopy data to understand observed trends in shape changes and connect them to molecular design rules of peptide-based gold nanoparticle synthesis.

3. Vaddi, Kiran, Karen Li, and Lilo D. Pozzo. "Metric geometry tools for automatic structure phase map generation." Digital Discovery 2.5 (2023): 1471-1483.

Introduces shape-based clustering of Small-Angle X-ray Scattering (SAXS) data and use it generate phase maps automatically from high-throughput samples of block-copolymers and blends.

## Best practices for applying to your data
1. Data-processing is needed as the methods used here are sensitive to data sampling. Theoretically, this should not be the case but in practice, we use techniques such as dynamic programming which works best if we have the unfirom sampling on the X-axis. This is often curcumvented by first fitting a spline to the data and using to make uniform samples. 
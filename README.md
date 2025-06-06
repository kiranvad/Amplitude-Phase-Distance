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

## Testing

A comprehensive test suite is available in the `testing/` directory. To run tests:

```bash
# Quick verification (no dependencies required)
cd testing && python test_basic_functionality.py

# Full test suite (requires pytest)
cd testing && python run_tests.py

# Or using pytest directly
cd testing && pytest tests/
```

For detailed testing instructions, see [`testing/TESTING_GUIDE.md`](testing/TESTING_GUIDE.md).


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

2. There are two versions to compute the distance. The default numpy version is from the original `fda_srsf` package and is the fastet and should be prefered. The torch version is an attempt to alleviate some of the issues with the numpy version. While the actual definition of distances computed in both versions are the same, they differ in how they approximate the correct warping function. The numpy version using dynamic programing is a *discrete* approach to obtain the warping function. In the torch method, we instead solve for the warping function by performing an optimization on the function space of warping functions under a particular metric. The mathematical details can be found in [this](https://arxiv.org/abs/2207.11141) paper. Currently the torch version is experimental and is sometimes known to throw nan values for distances. If that happens, try to re-run the program and it should then be able to converge to a better solution.
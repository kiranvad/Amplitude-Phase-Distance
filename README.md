# Amplitude-Phase-Distance
A light-weight repository to compute Amplitude Phase distance between two functions

## Installation

### Complete Installation (Recommended)

Install everything you need in one command:

```bash
pip install "git+https://github.com/kiranvad/Amplitude-Phase-Distance.git[complete]"
```

This installs the main package plus all optional dependencies (PyTorch, funcshape, warping).

### Basic Installation

```bash
pip install git+https://github.com/kiranvad/Amplitude-Phase-Distance.git
```

### Selective Installation

**With PyTorch support:**
```bash
pip install "git+https://github.com/kiranvad/Amplitude-Phase-Distance.git[torch]"
```

**With warping optimization:**
```bash
pip install "git+https://github.com/kiranvad/Amplitude-Phase-Distance.git[warping]"
```

**With development tools:**
```bash
pip install "git+https://github.com/kiranvad/Amplitude-Phase-Distance.git[dev]"
```

**Everything (complete + dev tools):**
```bash
pip install "git+https://github.com/kiranvad/Amplitude-Phase-Distance.git[all]"
```

### Development Installation

```bash
git clone https://github.com/kiranvad/Amplitude-Phase-Distance.git
cd Amplitude-Phase-Distance
pip install -e .
```

### Verify Installation

```python
import apdist
print(f"apdist version: {apdist.__version__}")

# Test basic functionality
import numpy as np
t = np.linspace(0, 1, 51)
f1 = np.sin(2 * np.pi * t)
f2 = np.sin(2 * np.pi * t + np.pi/4)
da, dp = apdist.AmplitudePhaseDistance(t, f1, f2)
print(f"Distance computation works: da={da:.4f}, dp={dp:.4f}")
```

Or run the included verification script:
```bash
python verify_installation.py
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

2. There are two versions to compute the distance. While the actual definition of distances computed in both versions are the same, they differ in how they approximate the correct warping function.:
    a. The default numpy version is from the original `fda_srsf` package and is the fastet and should be prefered. The numpy version using dynamic programing is a *discrete* approach to obtain the warping function.
    b. The torch version is an attempt to alleviate some of the issues with the numpy version. In the torch method, we instead solve for the warping function by performing an optimization on the function space of warping functions under a particular metric. The mathematical details can be found in [this](https://arxiv.org/abs/2207.11141) paper. We have observed that for in general torch and numpy version provide the same warping functions but the torch version is often better at warping complex functions with subtle features such as a small angle scattering curve in logspace. See this [notebook](/notebooks/01-saxs.ipynb) for more details.
    c. The torch version is non-determinstic since it uses a gradient descent approach and thus require large number of random restarts to converge to a solution. This can be controlled by the `n_restarts` parameter in the `AmplitudePhaseDistance` function.
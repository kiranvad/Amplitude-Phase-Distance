from setuptools import setup, find_packages
import os

# Import version from _version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'apdist', '_version.py')
    with open(version_file) as f:
        exec(f.read())
    return locals()['__version__']

setup(name="apdist",
      description="Amplitude-Phase distance between functions",
      version=get_version(),
      author='Kiran Vaddi',
      author_email='kiranvad@uw.edu',
      url='https://github.com/kiranvad/Amplitude-Phase-Distance',
      license='MIT',
      python_requires='>=3.8',
      install_requires=['numpy>=1.18.1',
                        'scipy',
                        'matplotlib',
                        'Cython',
                        'cffi',
                        'torch>=1.9.0',
                        'funcshape @ git+https://github.com/kiranvad/funcshape.git',
                        'warping @ git+https://github.com/kiranvad/warping.git'
                        ],
      extras_require = {
          'dev': [
              'pytest>=6.0',
              'pytest-cov>=2.10',
              'black',
              'isort',
              'flake8',
              'mypy',
          ],
          'all': [
              'pytest>=6.0',
              'pytest-cov>=2.10',
              'black',
              'isort',
              'flake8',
              'mypy',
          ],
      },
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows"
      ],
)
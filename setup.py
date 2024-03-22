from setuptools import setup,find_packages
import sys, os

setup(name="apdist",
      description="Amplitude-Phase distance between functions",
      version='1.0',
      author='Kiran Vaddi',
      author_email='kiranvad@uw.edu',
      license='MIT',
      python_requires='>=3.8',
      install_requires=['numpy>=1.18.1','scipy', 'matplotlib', 'Cython', 'cffi'],
      extras_require = {},
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
from setuptools import find_packages, setup

setup(
    name="benchmark",
    packages=find_packages(include=["benchmark"]),
    version="0.1.0",
    description="Sequence recovery benchmark",
    author="Rokas Petrenas, Chris Wood Lab, University of Edinburgh",
    license="MIT",
    test_suite="test",
    install_requires=['wheel','ampal','wget','numpy==1.19.5','pandas==1.2.0','scikit-learn==0.24.1','pathlib==1.0.1','matplotlib==3.3.3','click==7.1.2','scipy==1.6.0']
)

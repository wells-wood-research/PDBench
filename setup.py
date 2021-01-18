from setuptools import find_packages, setup

setup(
    name="benchmark",
    packages=find_packages(include=["benchmark"]),
    version="0.1.0",
    description="Sequence recovery benchmark",
    author="Rokas",
    license="MIT",
    test_suite="test",
)

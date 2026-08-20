from setuptools import find_packages, setup

packages = find_packages(include=["src", "src.*", "deepsat", "deepsat.*", "experiments", "experiments.*"])

setup(
    name="CanadaFireSat-Model",
    version="0.1.0",
    author="Hugo Porta",
    description="High-resolution wildfire forecasting using satellite imagery and environmental data",
    packages=packages,
)
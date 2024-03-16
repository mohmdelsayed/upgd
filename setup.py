from setuptools import setup, find_packages

setup(
    name="UPGD",
    version="1.0.0",
    description="Addressing Loss of Plasticity and Catastrophic Forgetting in Neural Networks",
    url="https://github.com/mohmdelsayed",
    author="Mohamed Elsayed",
    author_email="mohamedelsayed@ualberta.ca",
    packages=find_packages(exclude=["tests*"]),
    install_requires=['backpack-for-pytorch==1.3.0', 'HesScale==1.0.0'],
)

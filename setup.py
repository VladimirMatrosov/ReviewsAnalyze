from setuptools import setup, find_packages

setup(
    name="ReviewsAnalyzer",
    version=1.0,
    packages=find_packages(),
    install_requires=["tensorflow", "keras-tuner", "numpy", "matplotlib", "pandas"],
    author="Vladimir Matrosov",
    url="https://github.com/VladimirMatrosov/ReviewsAnalyze"
)
"""
Setup script for CalibratedActiveLearning package
"""

from setuptools import setup, find_packages

setup(
    name='calibrated-active-learning',
    version='0.1.0',
    description='Calibrated Active Learning using Empirical Likelihood',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'xgboost',
    ],
    python_requires='>=3.7',
)


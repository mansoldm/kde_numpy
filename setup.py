import setuptools
import pathlib
import os

with open('README.md', "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kde_numpy",  # Replace with your own username
    version="0.0.1",
    author="Martino Mansoldo",
    author_email="mansoldm@tcd.ie",
    description="Kernel Density Estimation in NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
    test_suite='tests',
    install_requires=[
        'tqdm',
        'numpy',
    ],
    python_requires=">=3.6",
)

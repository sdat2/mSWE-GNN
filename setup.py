from setuptools import setup
from typing import List

REQUIRED: List[str] = []


setup(
    name="mswegnn",
    description="The mSWE-GNN package: Graph Neural Networks for the Multiscale Shallow Water Equations",
    version="0.0.0",
    author_email="",
    author="",
    install_requires=REQUIRED,
    python_requires=">=3.9",
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Typing :: Typed",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=["mswegnn"],
    package_dir={
        "mswegnn": "mswegnn",
    },
)

import sys
from setuptools import setup, find_packages


setup_args = dict(
    name="som",
    version="0.0.1a",
    install_requires=[
        "numpy",
        "scipy"
    ],
    packages=find_packages(
        exclude=("demo",)
    ),
    url="https://github.com/ykatsu111/som.git"
)


if sys.version_info.major == 3:
    setup(
        use_2to3=True,
        **setup_args
    )
else:
    setup(
        **setup_args
    )

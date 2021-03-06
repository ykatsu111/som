import sys
from setuptools import setup, find_packages


setup_args = dict(
    name="som",
    version="0.0.1a",
    install_requires=[
        "numpy",
        "scipy"
    ],
    py_modules=(
        "som",
    ),
    url="https://github.com/ykatsu111/som.git"
)


if sys.version_info.major == 2:
    raise NotImplementedError(
        "python 2.x is not supported now."
    )
else:
    setup(
        **setup_args
    )

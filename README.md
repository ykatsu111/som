# Installation

In order to insall this module, please run below commands.

```
pip install git+https://github.com/ykatsu111/som.git
```

If you do not have permission, using `--user` option enables you to install on your home directory.

```
pip install --user git+https://github.com/ykatsu111/som.git
```

Please note that `2to3` is required If you install on Python3.x.

Another way to install is just to copy `som.py` to your working directory.  
In this way, this module only supports Python2.x.  
Moreover, please check `numpy` and `scipy` are installed on your python.

# Others

This module is an implimentation of [Nishiyama et al. (2007)](https://doi.org/10.1016/j.atmosres.2005.10.015).
However, this module uses square grid system, not hexagonal grid system which is used in Nishiyama et al.  
To learn Self-Organizing Maps (SOM), [Kohonen (1995)](https://doi.org/10.1007/978-3-642-56927-2) is probably one of the most famous text.

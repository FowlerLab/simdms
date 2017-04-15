simdms
======

simdms is a script for simulating deep mutational scanning data. These datasets are intended to be analyzed using [Enrich2](https://github.com/FowlerLab/Enrich2/). For more information or to cite simdms, please refer to [Enrich2: a statistical framework for analyzing deep mutational scanning data](http://biorxiv.org).

Dependencies
------------

simdms runs on Python 2.7 and requires the following packages:

* [NumPy](http://www.numpy.org/) version 1.10.4 or higher
* [SciPy](http://www.scipy.org/) version 0.16.0 or higher
* [pandas](http://pandas.pydata.org/) version 0.18.0 or higher
* [PyTables](http://www.pytables.org/) version 3.2.0 or higher

We recommend using a scientific Python distribution such as [Anaconda](https://store.continuum.io/cshop/anaconda/) or [Enthought Canopy](https://www.enthought.com/products/canopy/) to install and manage dependencies. PyTables may not be installed when using the default settings for your distribution. If you encounter errors, check that the `tables` module is present.

To use simdms, git clone or download the repository and run `python simdms.py config.json` from its root directory. simdms takes a json-formatted configuration file as the command line argument.

Questions?
----------

Please use the [GitHub Issue Tracker](https://github.com/FowlerLab/simdms/issues) to file bug reports or request features.

simdms was written by [Alan F Rubin](mailto:alan.rubin@wehi.edu.au) and Hannah Gelman.

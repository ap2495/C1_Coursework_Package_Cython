.. DualAutodifferentiation Package documentation master file, created by 
   sphinx-quickstart on Thu Dec  5 15:28:19 2024.

Dual Class Documentation
==============================================

- **Author**: Alexandr Prucha 
- **Project**: Autodifferentiation with Dual Numbers  
- **Version**: 0.1  
- **Python Version**: 3.12.3
- **License**: MIT 

Welcome to the documentation for the Dual Autodifferentiation Package for Cython! This package provides utilities for automatic differentiation.

Installation
------------

To install the package, download the wheel for your python version and run the following command in the Python shell:

.. code-block:: bash

   pip install ./dual_autodiff_x-0.0.1b2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

for python3.11 or:

.. code-block:: bash

   pip install ./dual_autodiff_x-0.0.1b2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

for python3.10.

Note: Separate installations of ``numpy`` and ``pytest`` are required.

In this section we provide detailed documentation for the modules and classes in the ``dual_autodiff_x`` package.


Dual Class Overview
===================

The class ``Dual_x`` generates Dual numbers from pairs of inputs. It also provides methods for performing basic calculations, and evaluating elementary functions.
Like its equivalen `Dual` class in the pure Python version of the package, ``Dual_x`` dynamically type-checks inputs and handles
them appropriately as scalars or arrays. For faster computation, the package also includes class ``Dual_x_array`` for specific use
on arrays, that does statically declares the variables as array-like. In terms of syntax, ``Dual_x`` and ``Dual_x_array`` are otherwise
identical, so we only include ``Dual_x`` here.

.. automodule:: dual_autodiff_x.dual
   :members:
   :special-members: __add__, __sub__, __mul__, __pow__
   :undoc-members:
   :show-inheritance:
   :no-index:
   :exclude-members: Dual_x_array



.. toctree::
   :maxdepth: 2
   :caption: Class Documentation
   :hidden:

   self



Jupyter Notebook Example
========================

Below is an example of a Jupyter notebook demonstrating the package's usage:

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   dual_autodiff_x


.. toctree::
   :maxdepth: 1
   :caption: Indices

   Index <genindex>
   Module Index <modindex>


Module Reference
----------------
.. toctree::
   :maxdepth: 1
   :caption: Summary

   Dual Autodiff_x Module <source/modules>

.. dolfin navier scipy documentation master file, created by
   sphinx-quickstart on Wed Jan  8 21:14:41 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dolfin_navier_scipy's documentation!
===============================================

The package *dolfin_navier_scipy (dns)* provides an interface between *scipy* and *FEniCS* in view of solving Navier-Stokes Equations. *FEniCS* is used to perform a Finite Element discretization of the equations. The assembled coefficients are exported as sparse matrices for use in *scipy*. Nonlinear and time-dependent parts are evaluated and assembled on demand. Visualization is done via the *FEniCS* interface to *paraview*.

Contents:

.. toctree::
  :maxdepth: 2

  Code


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


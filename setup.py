from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='dolfin_navier_scipy',
      version='1.0.0',
      description='A Scipy-Fenics interface for incompressible Navier-Stokes',
      license="GPLv3",
      long_description=long_description,
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/dolfin_navier_scipy",
      packages=['dolfin_navier_scipy'],  # same as name
      install_requires=['numpy', 'scipy',
                        'sadptprj_riclyap_adi'],  # ext packages dependencies
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GPLv3 License",
          "Operating System :: OS Independent",
          ]
      )

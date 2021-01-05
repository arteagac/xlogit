==============================================================================
xlogit: A Python package for GPU-accelerated estimation of mixed logit models.
==============================================================================

|Travis| |Coverage| |Docs| |PyPi| |License|

.. _Mixed Logit: https://xlogit.readthedocs.io/en/latest/api/mixed_logit.html
.. _Multinomial Logit: https://xlogit.readthedocs.io/en/latest/api/multinomial_logit.html

`Examples <https://xlogit.readthedocs.io/en/latest/examples.html>`__ | `Docs <https://xlogit.readthedocs.io/en/latest/index.html>`__ | `Installation <https://xlogit.readthedocs.io/en/latest/install.html>`__ | `API Reference <https://xlogit.readthedocs.io/en/latest/api/index.html>`__ | `Contributing <https://xlogit.readthedocs.io/en/latest/contributing.html>`__ | `Contact <https://xlogit.readthedocs.io/en/latest/index.html#contact>`__ 

Quick start
===========
The following example uses ``xlogit`` to estimate a mixed logit model for choices of electricity supplier (`See the data here <https://github.com/arteagac/xlogit/blob/master/examples/data/electricity_long.csv>`__). The parameters are:

* ``X``: 2-D array of input data (in long format) with choice situations as rows, and variables as columns
* ``y``: 1-D array of choices (in long format)
* ``varnames``: List of variable names that matches the number and order of the columns in ``X``
* ``alts``:  1-D array of alternative indexes or an alternatives list
* ``ids``:  1-D array of the ids of the choice situations
* ``panels``: 1-D array of ids for panel formation
* ``randvars``: dictionary of variables and their mixing distributions (``"n"`` normal, ``"ln"`` lognormal, ``"t"`` triangular, ``"u"`` uniform, ``"tn"`` truncated normal)

The current version of `xlogit` only supports input data in long format.

.. code-block:: python

    # Read data from CSV file
    import pandas as pd
    df = pd.read_csv("examples/data/electricity_long.csv")
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    X = df[varnames]
    y = df['choice']
    
    # Fit the model with xlogit
    from xlogit import MixedLogit
    model = MixedLogit()
    model.fit(X, y, 
              varnames,
              alts=df['alt'],
              ids=df['chid'],
              panels=df['id'],
              randvars={'pf': 'n','cl':'n','loc':'n','wk':'n','tod':'n','seas':'n'}, 
              n_draws=600)
    model.summary()


::

    Estimation with GPU processing enabled.
    Optimization terminated successfully.
    Estimation time= 5.2 seconds
    ---------------------------------------------------------------------------
    Coefficient              Estimate      Std.Err.         z-val         P>|z|
    ---------------------------------------------------------------------------
    pf                     -0.9996286     0.0331488   -30.1557541     9.98e-100 ***
    cl                     -0.2355334     0.0220401   -10.6865870      1.97e-22 ***
    loc                     2.2307891     0.1164263    19.1605300      5.64e-56 ***
    wk                      1.6251657     0.0918755    17.6887855      6.85e-50 ***
    tod                    -9.6067367     0.3112721   -30.8628296     2.36e-102 ***
    seas                   -9.7892800     0.2913063   -33.6047603     2.81e-112 ***
    sd.pf                   0.2357813     0.0181892    12.9627201      7.25e-31 ***
    sd.cl                   0.4025377     0.0220183    18.2819903      2.43e-52 ***
    sd.loc                  1.9262893     0.1187850    16.2166103      7.67e-44 ***
    sd.wk                  -1.2192931     0.0944581   -12.9083017      1.17e-30 ***
    sd.tod                  2.3354462     0.1741859    13.4077786      1.37e-32 ***
    sd.seas                -1.4200913     0.2095869    -6.7756668       3.1e-10 ***
    ---------------------------------------------------------------------------
    Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Log-Likelihood= -3888.413
    AIC= 7800.827
    BIC= 7847.493


For more examples of ``xlogit`` see `this Jupyter Notebook in Google Colab <https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb>`__. **Google Colab provides GPU resources for free**, which will significantly speed up your model estimation using ``xlogit``.

Quick install
=============
Install ``xlogit`` using ``pip`` as follows:

.. code-block:: bash

   pip install xlogit


.. hint::

   To enable GPU processing, you must install the `CuPy Python library <https://docs.cupy.dev/en/stable/install.html>`__.  When ``xlogit`` detects that CuPy is properly installed, it switches to GPU processing without any additional setup. If you use Google Colab, CuPy is usually installed by default.


For additional installation details check xlogit installation instructions at: https://xlogit.readthedocs.io/en/latest/install.html


No GPU? No problem
==================
``xlogit`` can also be used without a GPU. However, if you need to speed up your model estimation, there are several low cost and even free options to access cloud GPU resources. For instance:

- `Google Colab <https://colab.research.google.com>`_ offers free GPU resources for learning purposes with no setup required, as the service can be accessed using a web browser. Using xlogit in Google Colab is very easy as it runs out of the box without needing to install CUDA or CuPy, which are installed by default. For examples of xlogit running in Google Colab `see this link <https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb>`_.
- The `Google Cloud platform <https://cloud.google.com/compute/gpus-pricing>`_ offers GPU processing starting at $0.45 USD per hour for a NVIDIA Tesla K80 GPU with 4,992 CUDA cores.
- `Amazon Sagemaker <https://aws.amazon.com/ec2/instance-types/p2/>`_ offers virtual machine instances with the same TESLA K80 GPU at less than $1 USD per hour.

Benchmark
=========
As shown in the plots below, ``xlogit`` is significantly faster than existing estimation packages. Also, ``xlogit`` provides convenient scaling when the number of random draws increases. These results were obtained using a modest and low-cost NVIDIA GTX 1060 graphics card. More sophisticated graphics cards are expected to provide even faster estimation times. For additional details about this benchmark and for replication instructions check https://xlogit.readthedocs.io/en/latest/benchmark.html.

.. image:: https://raw.githubusercontent.com/arteagac/xlogit/master/examples/benchmark/results/time_benchmark_artificial.png
  :width: 300

.. image:: https://raw.githubusercontent.com/arteagac/xlogit/master/examples/benchmark/results/time_benchmark_apollo_biogeme.png
  :width: 300

Notes
=====
The current version allows estimation of:

- `Mixed Logit`_ with several types of mixing distributions (normal, lognormal, triangular, uniform, and truncated normal)
- `Mixed Logit`_ with panel data
- `Mixed Logit`_ with unbalanced panel data
- `Mixed Logit`_ with Halton draws
- `Multinomial Logit`_ models
- `Conditional logit <https://xlogit.readthedocs.io/en/latest/api/multinomial_logit.html>`_ models
- Weighed regression for all of the logit-based models

Contact
=======

If you have any questions, ideas to improve ``xlogit``, or want to report a bug, just open a `new issue in xlogit's GitHub repository <https://github.com/arteagac/xlogit/issues>`__ .

Citing ``xlogit``
=================
Please cite ``xlogit`` as follows:

    Arteaga, C., Park, J., Bhat, P., & Paz, A. (2021). xlogit: A Python package for GPU-accelerated estimation of mixed logit models. https://github.com/arteagac/xlogit
    
Or using BibTex as follows::

    @misc{xlogit,
        author = {Arteaga, Cristian and Park, JeeWoong and Bhat, Prithvi and Paz, Alexander},
        title = {{xlogit: A Python package for GPU-accelerated estimation of mixed logit models.}},
        url = {https://github.com/arteagac/xlogit},
        year = {2021}
    }


.. |Travis| image:: https://travis-ci.com/arteagac/xlogit.svg?branch=master
   :target: https://travis-ci.com/arteagac/xlogit

.. |Docs| image:: https://readthedocs.org/projects/xlogit/badge/?version=latest
   :target: https://xlogit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Coverage| image:: https://coveralls.io/repos/github/arteagac/xlogit/badge.svg?branch=master
   :target: https://coveralls.io/github/arteagac/xlogit?branch=master

.. |PyPi| image:: https://badge.fury.io/py/xlogit.svg
   :target: https://badge.fury.io/py/xlogit

.. |License| image:: https://img.shields.io/github/license/arteagac/xlogit
   :target: https://github.com/arteagac/xlogit/blob/master/LICENSE

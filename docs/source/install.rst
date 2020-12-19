Installation
============

xlogit requires Python 3.6 or later and it can be easily installed using ``pip`` as follows::

    pip install xlogit

This will automatically install the dependencies (i.e. `numpy <https://github.com/numpy/numpy>`_>=1.13.1 and `scipy <https://github.com/scipy/scipy>`_>=1.0.0). 

Enable GPU Processing
---------------------
By default, xlogit runs on the CPU. To enable GPU processing, it is necessary to additionally install the `CuPy <https://github.com/cupy/cupy>`_ Python package. When xlogit detects that CuPy is properly installed, it automatically switches to GPU processing without requiring any additional setup. To install CuPy you need:

1. Download and install the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`_.
2. Install the CuPy version that matches the installed CUDA Toolkit as described `in CuPy's docs <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_. 

.. hint::

   For instance, if you installed the `CUDA Toolkit v11.0 <https://developer.nvidia.com/cuda-11.0-download-archive>`_ , then you need to install the corresponding CuPy's version as follows::
    
    pip install cupy-cuda110


For additional details and troubleshooting of CuPy's installation see: https://docs.cupy.dev/en/stable/install.html


No GPU? No problem
------------------
xlogit can also be used without a GPU. However, if you need to speed up your model estimation, there are several low cost and even free options to access cloud GPU resources. For instance:

- `Google Colab <https://colab.research.google.com>`_ offers free GPU resources for learning purposes with no setup required, as the service can be accessed using a web browser. Using xlogit in Google Colab is very easy as it runs out of the box without needing to install CUDA or CuPy, which are installed by default. For examples of xlogit running in Google Colab `see this link <https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb>`_.
- The `Google Cloud platform <https://cloud.google.com/compute/gpus-pricing>`_ offers GPU processing starting at $0.45 USD per hour for a NVIDIA Tesla K80 GPU with 4,992 CUDA cores.
- `Amazon Sagemaker <https://aws.amazon.com/ec2/instance-types/p2/>`_ offers virtual machine instances with the same TESLA K80 GPU at less than $1 USD per hour.
=========
Benchmark
=========


Option 1: Quick benchmark in Google Colab
=========================================
This is the easiest way to execute the benchmark. Nothing needs to be installed, you just need a Gmail account to access Google Colab with GPU resources for free. This benchmark is limited to comparison of the Python tools (xlogit, pylogit, and biogeme) as Google Colab does not run R code. However, this quick benchmark is enough to demonstrate how fast ``xlogit`` is compared to existing tools. Also, it replicates several of the tables and figures in the paper.  To execute this benchmark, click the following link to open the benchmark source code in Google Colab and then select ``Runtime > Run all`` to run all the cells. Make sure a GPU Hardware Accelerator is being used by clicking ``Runtime > Change runtime type``. This benchmark should not take longer than 20 minutes. 


Option 2: Mini benchmark
========================
This is a minimal version of the full benchmark that can be executed in less than one hour. Executing this benchmark requires some basic knowledge in execution of commands in a Windows or Linux command line.

2.1 Requirements
----------------
* CUDA-enabled NVIDIA Graphics Card
* Windows or Linux Operating System
* Processor at least with 6 Cores (for apollo and biogeme)

.. hint::
   **Docker image available**. If you have a Linux machine, the easiest way to run the mini (and full) benchmark is using the Docker image "" in Docker's repository, which contains everything installed and the only requirement is to run the docker image as follows::

    docker run --gpus all xlogit-benchmark

   After running the benchmkark, al the results are saved in the ``benchmark/results`` folder. The ``Dockerfile`` used to create the ``xlogit-benchmark`` image can be found in the ``benchmark`` folder.

2.2 Installation steps
----------------------
Step 1. Setup Python tools
^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1.1 Install Python
"""""""""""""""""""""""
Miniconda provides a convenient toolset to install and work with Python libraries. Download miniconda (for Python 3) using the following links:

* For Windows: https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Windows-x86_64.exe
* For Linux: https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

After installing miniconda, open a command line in Linux or the Anaconda Prompt in Windows (``Start Menu > Anaconda > Anaconda Prompt``) and verify that the ``conda`` command is available by executing the following command and making sure no errors are shown::

    conda --version

Step 1.2 Setup a conda Python envinronment
""""""""""""""""""""""""""""""""""""""""""
Create and activate a conda environment (Python 3.8) for the benchmark by executing the following commands in a command line or Anaconda Prompt in Windows::

    conda create -n benchmark python=3.8
    conda activate bechmark

After running the previous commands, make sure that the ``python`` and ``pip`` commands are available and they use the correct Python version (3.8) by executing::

    python --version
    pip --version

.. warning::
   All the commands in the upcoming steps must be executed after activating the ``benchmark`` conda environment. Therefore, if you close your command line or Anaconda Prompt, you need to execute again ``conda activate benchmark`` before executing any command.

Step 1.3 Install the CuPy Python Package
""""""""""""""""""""""""""""""""""""""""
CuPy is required to use GPU processing in ``xlogit``. Follow these steps to install CuPy:

1. Install the CUDA Toolkit v11.0 for your operating system by downloading it from: https://developer.nvidia.com/cuda-11.0-download-archive. This may downgrade your installed NVIDIA driver so you may need to upgrade your driver again after running the benchmark.

2. Install the CUDA Toolkit in your ``benchmark`` conda environment by running the following command in the command line or Anaconda Prompt in Windows::

    conda install cudatoolkit==11.0.221

3. Run the following command to update ``pip`` as required by CuPy::

    python -m pip install -U setuptools pip

4. Install CuPy for CUDA Toolkit 11.0 by running in the command line or Anaconda Prompt::

    pip install cupy-cuda110

5. Verify that CuPy was properly installed by running the following command, which must run without showing any errors::

    python -c "import cupy"

.. hint::
   Although these instructions assume that you will use the CUDA Toolkit v11.0 and the associated CuPy version, you can install any other version of the CUDA Toolkit and CuPy that matches best your existing NVIDIA Driver. Check CuPy's installation instructions in `this link <https://docs.cupy.dev/en/stable/install.html>`__ for additional information or troubleshooting of CuPy's installation.

Step 1.4 Install Python packages for benchmark
""""""""""""""""""""""""""""""""""""""""""""""
In this step, ``xlogit``, ``pylogit``, and ``biogeme`` are installed. In your command line (or Anaconda Prompt in windows) navigate to the location of the provided ``benchmark`` folder using the ``cd`` (change directory) command (e.g. ``cd C:\User\Downloads\xlogit-benchmark\benchmark``) and then install the Python requirements using the following command::

    pip install -r requirements_python.txt
    pip install pylogit==0.2.2
    pip install biogeme==3.2.6

The ``biogeme`` Python package sometimes has issues during the initialization so reinstalling it helps avoiding future issues. To reinstall it, use the following commands::

    pip uninstall biogeme
    pip install biogeme==3.2.6 --no-cache-dir

Step 2. Setup R tools
^^^^^^^^^^^^^^^^^^^^^
Step 2.1 Install R v4.0
"""""""""""""""""""""""
You must use R version 4.0 (and not 3.6) for the benchmark as the installation of dependencies is easier with this version. 

* For Windows: Download R v4.0 from  https://cran.r-project.org/bin/windows/base/R-4.0.3-win.exe and follow the installation prompts. Make sure that R is available from the Anaconda Prompt by executing ``Rscript --version``. If this command does not run properly, you may need to add R's installation location to the Path envinronment variable as shown in the image in `this link <https://arteagac.github.io/images/other/add_environment_variable_win10.png>`__.

* For Linux: Depending on your distrubution, different instructions for installation of R v4.0 are available at https://docs.rstudio.com/resources/install-r/. Just make sure you select v4.0, instead of 3.6, which is the default suggested by the instructions. 
.. hint::
   For instance, if you use Ubuntu 20.04, you need to run the following commands to install Rv4.0::
   
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
    sudo apt-get update
    sudo apt-get -y install r-base libxml2-dev

Make sure that the ``Rscript`` command can be called from the command line (or Anaconda Prompt in Windows) by running the following command and checking that the correct version is shown::

    Rscript --version

Step 2.2 Install R packages
"""""""""""""""""""""""""""
This step installs the apollo and mlogit R packages. In your command line (or Anaconda Prompt in windows) navigate to the location of the provided ``benchmark`` folder using the ``cd`` (change directory) command (e.g. ``cd C:\User\Downloads\xlogit-benchmark\benchmark``) and then execute the command below. This command may require Administrator permissions so if you are in Windows Run the Anaconda Prompt as Administrator or if you are in Linux run this command as ``sudo``::

    Rscript requirements_r.R

Step 3. Run the mini benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, navigate to the location of the provided ``benchmark`` folder using the ``cd`` (change directory) command (e.g. ``cd C:\User\Downloads\xlogit-benchmark\benchmark``). Seconda, make sure that all the dependencies were properly installed by running the following command::

    python check_dependiencies.py

Finally, run the benchmark using the following command::

    python benchmark.py mini

The results of the benchmark are saved in the ``benchmark/results/`` folder.

Option 3: Full benchmark
========================
This is the full version of the benchmark that should take from 12 to 24 hours to run. Given that the full benchmark compares the performance of apollo and biogeme using up to 64 processor cores, a very powerful computer is needed for this benchmark.

3.1 Requirements
------------
* CUDA-enabled NVIDIA Graphics Card
* Windows or Linux Operating System
* Processor with at least with 64 Cores (for apollo and biogeme)

.. hint::
   **Docker image available**. If you have a Linux machine, the easiest way to run the mini (and full) benchmark is using the Docker image "" in Docker's repository, which contains everything installed and the only requirement is to run the docker image as follows::

    docker run --gpus all xlogit-benchmark

After running the benchmkark, al the results are saved in the ``benchmark/results`` folder. The ``Dockerfile`` used to create the ``xlogit-benchmark`` image can be found in the ``benchmark`` folder.

3.2 Installation steps
----------------------
Follow all the same steps as in the mini-benchmark (section 2.2) to install the dependencies. The only difference is the final command to execute the benchmark that must be in this case::

    python benchmark.py mini


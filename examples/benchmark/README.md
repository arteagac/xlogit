# Quick benchmark in Google Colab
This is the easiest way to execute the benchmark. Nothing needs to be installed, you just need a Gmail account to access Google Colab with GPU resources for free. However, this benchmark is limited to comparison of the Python tools (xlogit, pylogit, and biogeme) as Google Colab does not run R code. To execute this benchmark click the follow link and the select `Runtime > Run all` to run all the cells. This benchmark should not take longer than 20 minutes. Make sure a GPU Hardware Accelerator is being used by clicking `Runtime > Change runtime type`. This benchmark should provide a general idea of how fast xlogit is compared to pylogit and biogeme.

# Mini benchmark
This is a minimal version of the full benchmark that can be executed in less than one hour.

## Using Docker
If you have a Linux machine, the easiest way to run the mini (and full) benchmark is using the Docker image "" in Docker's repository, which contains everything installed and the only requirement is to run the docker image as follows:
`docker run --gpus all xlogit-benchmark`

To install Docker follow the instructions on their website: https://docs.docker.com/engine/install/ubuntu/#install-from-a-package

## Installing everything from scratch
Before running the benchmark, some tools need to be installed. Although all these tools are not necessary to use the `xlogit` package, the benchmark requires additional modules such as the pylogit and biogeme packages and the apollo and mlogit R packages for comparison. Also, python libraries to measure the memory usage such as "psutil" are required. The following are the steps to install the requirements:
### Step 1: Setup Python tools
#### Step 1.1 Install Python 3.7

Windows: `https://www.python.org/ftp/python/3.6.0/python-3.6.0-amd64.exe`  
Linux: `https://docs.conda.io/en/latest/miniconda.html#linux-installers`  
Once installed, make sure the `python` and `pip` commands are available in the command line. In case they are not, add the python installation bin folder to the PATH environment variable.

#### Step 1.2 Install CUDA Drivers and toolkit to enable GPU Processing
CuPy, the package that xlogit uses for GPU processing needs the CUDA Toolkit installed. Use conda as follows:  
`conda install cudatoolkit==11.0.221`

#### Step 1.3 Install Python dependencies
This is recommended by CuPy developers as expressed at https://docs.cupy.dev/en/stable/install.html   
`python -m pip install -U setuptools pip`  
### 2. Setup R tools
#### 2.1 Download and install R 4.0
Dowload link: https://cran.r-project.org/bin/windows/base/
Make sure Rscript command is available in the console. In case it is not, add the R installation bin folder to the PATH env. variable.  
## 3.2 Install R dependencies 
Administrator permissions are needed.  
`Rscript -e "install.packages('mlogit', repos='http://cran.us.r-project.org')"`
`install.packages("devtools")`
`require(devtools)`
`install_version("mlogit", version = "1.1-1")`
`install_version("apollo", version = "0.1.0")`
### 3. Run benchmark
`python benchmark.py mini`  
The results are saved in the 'results' folder and shown in console.

# Full benchmark
## Using Docker
If you have a Linux machine, the easiest way to run the mini (and full) benchmark is using the Docker image "" in Docker's repository, which contains everything installed and the only requirement is to run the docker image as follows:
`docker run --gpus all xlogit-benchmark`

## Installing everything from scratch

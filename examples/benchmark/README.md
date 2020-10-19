Before running the benchmark, some tools need to be installed. Even though all these tools are not necessary for the `xlogit` library, the benchmark requires additional modules such as the pylogit and R:mlogit libraries for comparison of results. Also, python libraries to measure the memory usage such as "psutil" are required.
# 1. Install Python tools
## 1.1 Install Python 3.6, 3.7, or 3.8
Windows: `https://www.python.org/ftp/python/3.6.0/python-3.6.0-amd64.exe`  
Linux: `https://docs.conda.io/en/latest/miniconda.html#linux-installers`  
Once installed, make sure the `python` and `pip` commands are available in the command line. In case they are not, add the python installation bin folder to the PATH env. variable.

## 1.2 Install requirements for benchmark
Install the requiriments in the file "requirements_bench.txt" in this same folder. Notice that these requirements are different to the `xlogit` requirements as some extra libraries are necessary for the benchmark.  
`pip install -r requirements_bench.txt`

# 2. Install CuPy for GPU Processing
CuPy is the library that `xlogit` uses for GPU processing. The following are summarized instructions based on the instructions from CuPy's documentation:   https://docs.cupy.dev/en/stable/install.html
## 2.1 Update the Python setup tools
This is recommended by CuPy developers as expressed at https://docs.cupy.dev/en/stable/install.html   
`python -m pip install -U setuptools pip`  
## 2.2 Download and install CUDA toolkit
Download CUDA toolkit v11 from https://developer.nvidia.com/cuda-downloads and follow their instructions for installation.  
## 2.3 Install cupy. 
Install cupy depending on the version of CUDA toolkit selected. For instance, for v11 the command would be:  
`pip install cupy-cuda110`  
For other versions see: https://docs.cupy.dev/en/stable/install.html  

# 3. Install R tools
## 3.1 Download and install R
Dowload link: https://cran.r-project.org/bin/windows/base/
Make sure Rscript command is available in the console. In case it is not, add the R installation bin folder to the PATH env. variable.  
## 3.2 Install the mlogit package 
Administrator permissions are needed.  
`Rscript -e "install.packages('mlogit', repos='http://cran.us.r-project.org')"`

# 4. Run benchmark
`python benchmark.py`  
The results are saved in the 'results' folder and shown in console.

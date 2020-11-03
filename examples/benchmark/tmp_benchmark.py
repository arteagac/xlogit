"""
curl -L -O https://github.com/arteagac/xlogit/archive/master.zip
unzip master.zip
cd xlogit-master/examples/benchmark/
pip3 install -r requirements_bench.txt
python3 -m pip install -U setuptools pip
pip3 install cupy-cuda102

This file executes the benchmark. Check the README.md file in this folder
to make sure all the requirments are satisfied.
"""

import os
from tools import init_profiler_output_file
import cupy as cp
if cp.asnumpy(cp.array([1, 2]).dot(cp.array([1, 2]))) == 5:
    print("Cupy is installed and properly configured")
# ==========================================
# pylogit and mlogit benchmark
# ==========================================
init_profiler_output_file()


def profile_range_draws(command, r_draws, dataset, usegpu=False):
    print("\n\n=== "+dataset+" dataset. "+command.split()[1] +
          ('(using GPU)' if usegpu else '')+" ===")
    print("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
    for r in range(1, r_draws+1):
        os.system("{} {} {} {} prof".format(command, r*100, dataset, usegpu*1))


def print_estimates(command, n_draws, dataset):
    print("\n\n=== "+dataset+" dataset. "+command.split()[1]+" ===")
    os.system("{} {} {} {} estim".format(command, n_draws, dataset, 0))


r_draws = 30
# Run profiling
profile_range_draws("python3 xlogit_run.py", r_draws, "artificial", True)
profile_range_draws("python3 xlogit_run.py", r_draws, "electricity", True)

"""
=== artificial dataset. xlogit_run.py(using GPU) ===
Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.
   100    0.88    -2276.43   0.393   0.114 True
   200    1.20    -2276.83   0.415   0.228 True
   300    1.43    -2278.06   0.409   0.341 True
   400    1.75    -2279.18   0.441   0.454 True
   500    2.11    -2277.76   0.441   0.567 True
   600    2.28    -2277.80   0.443   0.681 True
   700    2.66    -2278.16   0.503   0.794 True
   800    2.93    -2278.40   0.505   0.907 True
   900    3.08    -2277.84   0.505   1.020 True
  1000    3.52    -2278.20   0.507   1.134 True
  1100    3.74    -2278.31   0.508   1.247 True
  1200    3.93    -2278.45   0.504   1.360 True
  1300    4.15    -2278.17   0.503   1.473 True
  1400    4.52    -2278.82   0.629   1.586 True
  1500    5.04    -2278.10   0.629   1.700 True
  1600    5.24    -2278.12   0.633   1.813 True
  1700    5.19    -2278.02   0.631   1.926 True
  1800    5.37    -2278.19   0.652   2.039 True
  1900    5.69    -2278.17   0.682   2.153 True
  2000    5.70    -2278.20   0.717   2.266 True
  2100    6.41    -2278.32   0.738   2.379 True
  2200    6.50    -2278.41   0.819   2.492 True
  2300    6.57    -2278.03   0.804   2.606 True
  2400    6.97    -2278.26   0.905   2.719 True
  2500    7.62    -2278.23   0.937   2.832 True
  2600    7.67    -2278.32   0.971   2.945 True
  2700    7.89    -2278.17   0.938   3.059 True
  2800    7.92    -2278.49   1.018   3.172 True
  2900    8.42    -2278.32   1.075   3.285 True
  3000    8.58    -2278.04   1.034   3.398 True


=== electricity dataset. xlogit_run.py(using GPU) ===
Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.
   100    2.33    -3932.66   0.402   0.079 True
   200    1.76    -3925.33   0.383   0.156 True
   300    2.70    -3914.24   0.384   0.233 True
   400    3.14    -3893.06   0.387   0.311 True
   500    4.20    -3898.36   0.397   0.388 True
   600    4.06    -3887.71   0.397   0.466 True
   700    4.38    -3886.38   0.396   0.543 True
   800    5.40    -3882.11   0.392   0.621 True
   900    6.57    -3886.82   0.396   0.698 True
  1000    6.77    -3889.53   0.412   0.776 True
  1100    7.03    -3889.44   0.413   0.853 True
  1200    8.62    -3889.39   0.411   0.930 True
  1300    9.21    -3887.68   0.410   1.008 True
  1400    9.98    -3885.77   0.427   1.085 True
  1500   10.86    -3887.87   0.413   1.163 True
  1600   11.12    -3887.03   0.412   1.240 True
  1700   11.30    -3877.22   0.412   1.318 True
  1800   11.60    -3881.16   0.410   1.395 True
  1900   11.52    -3880.56   0.409   1.473 True
  2000   13.47    -3886.35   0.437   1.550 True
  2100   15.91    -3879.89   0.441   1.628 True
  2200   14.11    -3881.97   0.445   1.705 True
  2300   14.61    -3880.13   0.444   1.783 True
  2400   16.79    -3881.29   0.444   1.860 True
  2500   17.95    -3884.49   0.460   1.937 True
  2600   16.79    -3885.20   0.439   2.015 True
  2700   18.84    -3878.07   0.437   2.092 True
  2800   20.20    -3880.63   0.440   2.170 True
  2900   16.71    -3882.89   0.439   2.247 True
  3000   19.03    -3881.23   0.443   2.325 True

CPU:       Topology: 6x Single Core (4-Die) 
model: AMD EPYC (with IBPB) bits: 64 type: MCM SMP L2 cache: 3072 KiB 
Speed: 2500 MHz min/max: N/A Core speeds (MHz): 
1: 2500 2: 2500 3: 2500 4: 2500 5: 2500 6: 2500 

ubuntu@104-171-200-65:~/xlogit-master/examples/benchmark$ nvidia-smi
Mon Nov  2 23:38:54 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro RTX 6000     On   | 00000000:14:00.0 Off |                  Off |
| 33%   30C    P2    54W / 260W |    734MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     16197      C   python3                           731MiB |
+-----------------------------------------------------------------------------+

"""

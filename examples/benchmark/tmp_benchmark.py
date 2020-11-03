"""
curl -L -O https://github.com/arteagac/xlogit/archive/master.zip
unzip master.zip
cd xlogit/examples/benchmark
pip3 install benchmark_requirements.py
python -m pip install -U setuptools pip
pip3 install cupy-cuda110

This file executes the benchmark. Check the README.md file in this folder
to make sure all the requirments are satisfied.
"""

import os
from tools import init_profiler_output_file

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

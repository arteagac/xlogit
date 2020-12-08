"""
This file executes the benchmark. Check the README.md file in this folder
to make sure all the requirments are satisfied.
"""

import os
from tools import init_profiler_output_files, log
import cupy as cp
import sys

init_profiler_output_files()
mini = len(sys.argv) == 2 and sys.argv[1] == 'mini'

# ==========================================
# pylogit and mlogit benchmark
# ==========================================
if cp.asnumpy(cp.array([1, 2]).dot(cp.array([1, 2]))) == 5:
    print("Cupy is installed and properly configured")


def profile_range_draws(command, r_draws, dataset, usegpu=False):
    log("\n\n=== "+dataset+" dataset. "+command.split()[1] +
        ('(using GPU)' if usegpu else '')+" ===")
    log("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
    for r in range(1, r_draws+1):
        os.system("{} {} {} {} prof".format(command, r*100, dataset, usegpu*1))


def profile_range_draws_and_cores(command, r_draws, r_cores):
    log("\n\n=== artificial dataset. "+command.split()[1]+" ===")
    for n_draws in r_draws:
        for n_cores in r_cores:
            os.system("{} {} {}".format(command, n_draws, n_cores))


def print_estimates(command, n_draws, dataset):
    log("\n\n=== "+dataset+" dataset. "+command.split()[1]+" ===")
    os.system("{} {} {} {} estim".format(command, n_draws, dataset, 0))


r_draws = 4 if mini else 15

# Run profiling
profile_range_draws("python xlogit_run.py", r_draws, "artificial", True)
profile_range_draws("python xlogit_run.py", r_draws, "artificial")
profile_range_draws("python pylogit_run.py", r_draws, "artificial")
profile_range_draws("Rscript mlogit_run.R", r_draws, "artificial")
profile_range_draws("python xlogit_run.py", r_draws, "electricity", True)
profile_range_draws("python xlogit_run.py", r_draws, "electricity")
profile_range_draws("python pylogit_run.py", r_draws, "electricity")
profile_range_draws("Rscript mlogit_run.R", r_draws, "electricity")

# Print estimates
print_estimates("python xlogit_run.py", 400, "artificial")
print_estimates("python pylogit_run.py", 400, "artificial")
print_estimates("Rscript mlogit_run.R", 400, "artificial")
print_estimates("python xlogit_run.py", 400, "electricity")
print_estimates("python pylogit_run.py", 400, "electricity")
print_estimates("Rscript mlogit_run.R", 400, "electricity")


# ==========================================
# apollo and biogeme benchmark
# ==========================================
if mini:
    r_draws = [100, 200, 300]
    r_cores = [2, 4]
else:
    r_draws = [100, 500, 1000, 1500]
    r_cores = [16, 32, 64]

profile_range_draws_and_cores("python biogeme_run.py", r_draws, r_cores)
os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Avoids segfault error
profile_range_draws_and_cores("Rscript apollo_run.R", r_draws, r_cores)


# ==========================================
# plot results
# ==========================================
os.system("python plot_results.py"+" mini" if mini else "")

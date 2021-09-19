"""
This file executes the benchmark. Check the README.md file in this folder
to make sure all the requirments are satisfied.
"""

import os
from tools import init_profiler_output_files, log
import sys
sys.path.append("../../")  # Path of xlogit library root folder.
from xlogit import MixedLogit

MixedLogit.check_if_gpu_available()
mini = len(sys.argv) == 2 and sys.argv[1] == 'mini'
init_profiler_output_files()


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


# ==========================================
# pylogit and mlogit benchmark
# ==========================================
if mini:
    r_draws = 4
else:
    r_draws = 15

# Run profiling
log("\n\n********* PYLOGIT AND MLOGIT BENCHMARK *********")
profile_range_draws("python xlogit_run_batch.py", r_draws, "artificial", True)
profile_range_draws("python xlogit_run_batch.py", r_draws, "artificial")
profile_range_draws("python pylogit_run.py", r_draws, "artificial")
profile_range_draws("Rscript mlogit_run.R", r_draws, "artificial")
profile_range_draws("Rscript gmnl_run.R", r_draws, "artificial")
profile_range_draws("python xlogit_run_batch.py", r_draws, "electricity", True)
profile_range_draws("python xlogit_run_batch.py", r_draws, "electricity")
profile_range_draws("python pylogit_run.py", r_draws, "electricity")
profile_range_draws("Rscript mlogit_run.R", r_draws, "electricity")
profile_range_draws("Rscript gmnl_run.R", r_draws, "electricity")

# Print estimates
n_draws = 1500
log("\n\n********* ESTIMATES (COEFF AND STD.ERR.)*********")
print_estimates("python xlogit_run_batch.py", n_draws, "artificial")
print_estimates("Rscript gmnl_run.R", n_draws, "artificial")
print_estimates("Rscript mlogit_run.R", n_draws, "artificial")
print_estimates("python xlogit_run_batch.py", n_draws, "electricity")
print_estimates("Rscript gmnl_run.R", n_draws, "electricity")
print_estimates("Rscript mlogit_run.R", n_draws, "electricity")



# ==========================================
# apollo and biogeme benchmark
# ==========================================
if mini:
    r_draws = [100, 200, 300]
    r_cores = [2, 4]
else:
    r_draws = [100, 500, 1000, 1500]
    r_cores = [16, 32, 64]
log("\n\n********* APOLLO AND BIOGEME BENCHMARK *********")
profile_range_draws_and_cores("python biogeme_run.py", r_draws, r_cores)
os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Avoids segfault error
profile_range_draws_and_cores("Rscript apollo_run.R", r_draws, r_cores)
profile_range_draws_and_cores("Rscript mixl_run.R", r_draws, r_cores)

# ==========================================
# plot results
# ==========================================
os.system("python plot_results.py"+" mini" if mini else "")


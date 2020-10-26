import os

output_file = "results/benchmark_results_apollo_biogeme.csv"

if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, 'a') as fw:
    fw.write("library,draws,cores,time,loglik\n")


def profile_range_draws_and_cores(command, r_draws, r_cores):
    print("\n\n=== artificial dataset. "+command.split()[1]+" ===")
    for n_draws in r_draws:
        for n_cores in r_cores:
            os.system("{} {} {}".format(command, n_draws, n_cores))


r_draws = [100, 500, 1000, 1500]
r_cores = [16, 32, 64]


profile_range_draws_and_cores("python biogeme_run.py", r_draws, r_cores)
os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Avoids segfault error
profile_range_draws_and_cores("Rscript apollo_run.R", r_draws, r_cores)

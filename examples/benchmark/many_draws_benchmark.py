import os
import sys
from tools import init_profiler_output_files, log


def profile_range_draws(command, r_draws, dataset, usegpu=False):
    log("\n\n=== "+dataset+" dataset. "+command.split()[1] +
        ('(using GPU)' if usegpu else '')+" ===")
    log("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
    for r in r_draws:
        os.system(f"{command} {r} {dataset} {usegpu*1} prof")


init_profiler_output_files()
r_draws_gpu = [2000, 3000, 4000, 5000]
r_draws = r_draws_gpu + [10000, 15000]
profile_range_draws("python xlogit_run.py", r_draws_gpu, "artificial", True)
profile_range_draws("python xlogit_run.py", r_draws, "artificial", False)

r_draws_gpu = [2000, 3000, 4000, 5000, 7500]
r_draws = r_draws_gpu + [10000, 15000, 20000]
profile_range_draws("python xlogit_run.py", r_draws_gpu, "electricity", True)
profile_range_draws("python xlogit_run.py", r_draws, "electricity", False)


# ==========================================
# Plot results
# ==========================================
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams.update({'font.size': 14,
                            'axes.spines.right': False,
                            'axes.spines.top': False})
markers = ['o', 'x']
colors = ["#e41a1c", "#ff7f00"]
lines = ['--', '-']

df = pd.read_csv("results/many_draws_benchmark_results.csv")

libs = ['xlogit_gpu', 'xlogit']
datasets = ['artificial', 'electricity']
memtypes = ["gpu", "ram"]

xticks = {'artificial': [2000, 5000, 10000, 15000],
           'electricity': [2000, 5000, 7500, 10000, 15000, 20000]}

def plot_memory_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    plt.figure()
    leg = []
    for i, lib in enumerate(libs):
        memtype = "gpu" if "gpu" in lib else "ram"
        d = dfe[dfe.library == lib][["draws", memtype]].values.T
        plt.plot(d[0], d[1], marker=markers[i], c=colors[i],
                 linestyle=lines[i])
        leg.append(f"{lib} ({memtype.upper()})")
    plt.legend(leg)
    plt.xlabel("Random draws")
    plt.ylabel("Memory usage (GB)")
    plt.xticks(xticks[dataset])
    plt.title(f"Memory usage of xlogit ({dataset} dataset)")
    plt.savefig(f"results/memory_many_draws_{dataset}", dpi=300)
    plt.show()

def plot_time_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    plt.figure()
    for i, lib in enumerate(libs):
        d = dfe[dfe.library == lib][["draws", "time"]].values.T
        plt.plot(d[0], d[1], marker=markers[i], c=colors[i])
    plt.legend(libs)
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.xticks(xticks[dataset])
    plt.title("Estimation time ("+dataset+" dataset)")
    plt.savefig(f"results/time_many_draws_{dataset}", dpi=300)
    plt.show()
    
plot_memory_benchmark("electricity")
plot_memory_benchmark("artificial")
plot_time_benchmark("electricity")
plot_time_benchmark("artificial")


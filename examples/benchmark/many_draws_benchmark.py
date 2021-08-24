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

r_draw_artif = [1000, 5000, 25000, 50000]
profile_range_draws("python xlogit_run_batch.py", r_draw_artif, "artificial", True)
profile_range_draws("python xlogit_run_batch.py", r_draw_artif, "artificial", False)

r_draws_elec = [5000, 50000, 250000, 500000]
profile_range_draws("python xlogit_run_batch.py", r_draws_elec, "electricity", True)
profile_range_draws("python xlogit_run_batch.py", r_draws_elec, "electricity", False)


# ==========================================
# Plot results
# ==========================================
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.ticker as tick

matplotlib.rcParams.update({'font.size': 14, 'axes.spines.right': False, 'axes.spines.top': False})
markers = ['o', 'x']
colors = ["#e41a1c", "#ff7f00"]
lines = ['--', '-']

df = pd.read_csv("results/many_draws_benchmark_results.csv")

libs = ['xlogit_gpu', 'xlogit']
datasets = ['artificial', 'electricity']
memtypes = ["gpu", "ram"]

xticks = {'artificial': r_draw_artif,
           'electricity': r_draws_elec}

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
    plt.xticks(ticks=xticks[dataset], labels=[f"{x//1000}k" for x in xticks[dataset]])
    plt.title(f"Memory usage ({dataset} dataset)")
    plt.tight_layout()
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
    plt.ylabel("Time (Minutes)")
    plt.xticks(ticks=xticks[dataset], labels=[f"{x//1000}k" for x in xticks[dataset]])
    plt.title("Estimation time ("+dataset+" dataset)")
    plt.gca().yaxis.set_major_formatter(tick.FuncFormatter(lambda val, pos: int(val//60)))
    plt.tight_layout()
    plt.savefig(f"results/time_many_draws_{dataset}", dpi=300)
    plt.show()
    
plot_memory_benchmark("electricity")
plot_memory_benchmark("artificial")
plot_time_benchmark("electricity")
plot_time_benchmark("artificial")


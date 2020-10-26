import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


matplotlib.rcParams.update({'font.size': 14,
                            'axes.spines.right': False,
                            'axes.spines.top': False})
markers = ['s', '|', '^', 'x', 'o']
lines = ['-', '--']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ==========================================
# pylogit and mlogit benchmark
# ==========================================
df = pd.read_csv("results/benchmark_results.csv")

libs = ['pylogit', 'mlogit', 'xlogit', 'xlogit_gpu']


def plot_memory_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    plt.figure()
    for i, lib in enumerate(libs):
        d = dfe[dfe.library == lib][["draws", "ram"]].values.T
        plt.plot(d[0], d[1], marker=markers[i])
    d = dfe[dfe.library == "xlogit_gpu"][["draws", "gpu"]].values.T
    plt.plot(d[0], d[1], marker=markers[-1])
    plt.legend([i + " (RAM)" for i in libs] + ["xlogit_gpu (GPU)"])
    plt.xlabel("Random draws")
    plt.ylabel("Memory usage (GB)")
    plt.title("Memory usage ("+dataset+" dataset)")
    plt.savefig("results/memory_benchmark_"+dataset, dpi=300)
    plt.show()


def plot_time_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    plt.figure()
    for i, lib in enumerate(libs):
        d = dfe[dfe.library == lib][["draws", "time"]].values.T
        plt.plot(d[0], d[1], marker=markers[i])
    plt.legend(libs)
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.title("Estimation time ("+dataset+" dataset)")
    plt.savefig("results/time_benchmark_"+dataset, dpi=300)
    plt.show()


plot_memory_benchmark("electricity")
plot_memory_benchmark("artificial")

plot_time_benchmark("electricity")
plot_time_benchmark("artificial")


# ==========================================
# apollo and biogeme benchmark
# ==========================================
r_draws = [100, 500, 1000, 1500]
r_cores = [16, 32, 64]
dfab = pd.read_csv("results/benchmark_results_apollo_biogeme.csv")

# Add xlogit time data to benchmark results for apollo and biogeme
for xl in ['xlogit', 'xlogit_gpu']:
    for draws in r_draws:
        d = df[(df.library == xl) & (df.draws == draws) &
               (df.dataset == "artificial")]
        dfab.loc[len(dfab)] = [xl, draws, 0,
                               d.time.values[0], d.loglik.values[0]]


def plot_time_benchmark_apollo_biogeme(df):
    plt.figure(figsize=(12, 7))
    for li, lib in enumerate(['apollo', 'biogeme']):
        for m, cores in enumerate(r_cores):
            idx = (df.library == lib) & (df.cores == cores)
            d = df[idx][["draws", "time"]].values.T
            plt.plot(d[0], d[1], marker=markers[m], linestyle=lines[li],
                     c=colors[m], label="{} {} cores".format(lib, cores))
    for li, lib in enumerate(['xlogit_gpu', 'xlogit']):
        idx = (df.library == lib)
        d = df[idx][["draws", "time"]].values.T
        plt.plot(d[0], d[1], marker=markers[3+li], c=colors[3], label=lib)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3)
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.title("Estimation time (artificial dataset)")
    plt.tight_layout()
    plt.savefig("results/time_benchmark_apollo_biogeme", dpi=300)
    plt.show()


plot_time_benchmark_apollo_biogeme(dfab)

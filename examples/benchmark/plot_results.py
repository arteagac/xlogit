import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd
from tools import log

matplotlib.rcParams.update({'font.size': 14,
                            'axes.spines.right': False,
                            'axes.spines.top': False})
markers = ['d', 's', 'x', 'o', '^', '|']
colors = ["#377eb8", "#4daf4a", "#ff7f00", "#e41a1c", "#984ea3",
          "#a4c500"]

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
        plt.plot(d[0], d[1], marker=markers[i], c=colors[i])
    d = dfe[dfe.library == "xlogit_gpu"][["draws", "gpu"]].values.T
    plt.plot(d[0], d[1], marker=markers[3], c=colors[3],
             linestyle="--")
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
        plt.plot(d[0], d[1], marker=markers[i], c=colors[i])
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
mini = len(sys.argv) == 2 and sys.argv[1] == 'mini'
if mini:
    r_draws = [100, 200, 300]
    r_cores = [2, 4]
else:
    r_draws = [100, 500, 1000, 1500]
    r_cores = [16, 32, 64]

lines = [':', '--', '-']
dfab = pd.read_csv("results/benchmark_results_apollo_biogeme.csv")

# Add xlogit time data to benchmark results for apollo and biogeme
for lib in libs:
    for draws in r_draws:
        d = df[(df.library == lib) & (df.draws == draws) &
               (df.dataset == "artificial")]
        dfab.loc[len(dfab)] = [lib, draws, 0,
                               d.time.values[0], d.loglik.values[0]]


def plot_time_benchmark_apollo_biogeme(df):
    plt.figure(figsize=(8, 7))
    for li, lib in enumerate(['apollo', 'biogeme']):
        for c, cores in enumerate(r_cores):
            idx = (df.library == lib) & (df.cores == cores)
            d = df[idx][["draws", "time"]].values.T
            plt.plot(d[0], d[1], marker=markers[-li - 1],
                     linestyle=lines[c], c=colors[-li - 1],
                     label="{} {} cores".format(lib, cores))
    for li, lib in enumerate(libs):
        if lib == "xlogit" or lib == "xlogit_gpu":
            idx = (df.library == lib)
            d = df[idx][["draws", "time"]].values.T
            plt.plot(d[0], d[1], marker=markers[li], c=colors[li],
                     label=lib)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
               ncol=3)
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.title("Estimation time (artificial dataset)")
    plt.tight_layout()
    plt.savefig("results/time_benchmark_apollo_biogeme", dpi=300)
    plt.show()


plot_time_benchmark_apollo_biogeme(dfab)

# ==========================================
# comparison table
# ==========================================
# Keep only data for 64 cores
dfc = dfab.copy()
dfc = dfc.drop(dfc[(dfc.library == "apollo") & (dfc.cores != 64)].index)
dfc = dfc.drop(dfc[(dfc.library == "biogeme") & (dfc.cores != 64)].index)
dfc = dfc.drop(['cores', 'loglik'], axis=1)  # Keep only time
# Reshape to have time as columns
dfc = dfc.pivot(index='library', columns='draws', values='time')

# Compute in comparison against xlogit_gpu
for draws in r_draws:
    col = dfc[draws]
    dfc['c'+str(draws)] = col.values/col[col.index == "xlogit_gpu"].values
dfc['cavg'] = dfc[['c'+str(i) for i in r_draws]].values.mean(axis=1)
dfc = dfc.round(1)

# Print in a nice format
log("{:12} {:^28} {:^37}".format("", "Estimation time",
                                 "Compared to xlogit_gpu"))
c = dfc.columns.values
log("{:12} {:6} {:6} {:6} {:6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
    "library", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], "c_avg"))

for row in dfc.iterrows():
    c = row[1].values
    log("{:12} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6}".format(
        row[0], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]))

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

df = pd.read_csv("results/profiling_results.csv")

libs = ['pylogit', 'mlogit', 'xlogit', 'xlogit_gpu']
matplotlib.rcParams.update({'font.size': 14,
                            'axes.spines.right': False,
                            'axes.spines.top': False})
markers = ['s', '|', '^', 'x', 'o']

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

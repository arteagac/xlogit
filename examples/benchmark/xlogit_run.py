import numpy as np
import pandas as pd
from tools import Profiler, curr_ram, log
import sys
# sys.path.append("../../")  # Path of xlogit library root folder.
from xlogit import MixedLogit
from xlogit import device

data_folder = "../data/"
if len(sys.argv) == 5:  # If CLI arguments provided
    n_draws, dataset = int(sys.argv[1]), sys.argv[2]
    use_gpu, profile = sys.argv[3] == '1', sys.argv[4] == 'prof'
else:  # Default in case CLI arguments not provided
    n_draws, dataset, use_gpu, profile = 100, "artificial", False, False
# ==== Electricity dataset
if dataset == "electricity":
    df = pd.read_csv(data_folder+"electricity_long.csv")
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    X = df[varnames].values
    y = df['choice'].values
    randvars = {'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                'tod': 'n', 'seas': 'n'}
    alts = [1, 2, 3, 4]
    panels = df.id.values

# ==== Artificial dataset
if dataset == "artificial":
    df = pd.read_csv(data_folder+"artificial_long.csv")
    varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr',
                'emipp', 'nonsig1', 'nonsig2', 'nonsig3']
    X = df[varnames].values
    y = df['choice'].values
    randvars = {'meals': 'n', 'petfr': 'n', 'emipp': 'n'}
    alts = [1, 2, 3]
    panels = None

if not use_gpu:
    device.disable_gpu_acceleration()
if profile:
    ini_ram = curr_ram()
    profiler = Profiler().start(measure_gpu_mem=use_gpu)

np.random.seed(0)
model = MixedLogit()
model.fit(X, y, varnames, alts=alts, n_draws=n_draws,
          panels=panels, verbose=0, randvars=randvars)

if profile:
    ellapsed, max_ram, max_gpu = profiler.stop()
    log("{:6} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {}"
        .format(n_draws, ellapsed, model.loglikelihood,
                max_ram - ini_ram, max_gpu, model.convergence))
    profiler.export('xlogit'+('_gpu' if use_gpu else ''), dataset,
                    n_draws, ellapsed, model.loglikelihood, max_ram - ini_ram,
                    max_gpu, model.convergence)

if not profile:
    log("Variable    Estimate   Std.Err.")
    for i in range(len(model.coeff_names)):
        log("{:9}  {:9.5}  {:9.5}".format(model.coeff_names[i][:8],
                                          model.coeff_[i], model.stderr[i]))
    log("Log.Lik:   {:9.2f}".format(model.loglikelihood))

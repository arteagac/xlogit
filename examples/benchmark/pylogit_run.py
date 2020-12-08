import numpy as np
import pandas as pd
from tools import Profiler, curr_ram
import sys
import pylogit as pl
import io
from collections import OrderedDict
from tools import log


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
    alt_id_col = "alt"
    obs_id_col = "chid"
    choice_col = "choice"
    mixing_id_col = "id"
    mixing_vars = varnames
    spec, spec_names = OrderedDict(), OrderedDict()
    for col in varnames:
        df[col] = df[col].astype(float)
        spec[col] = [[1, 2, 3, 4]]
        spec_names[col] = [col]


# ==== Artificial dataset
if dataset == "artificial":
    df = pd.read_csv(data_folder+"artificial_long.csv")
    varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr',
                'emipp', 'nonsig1', 'nonsig2', 'nonsig3']
    alt_id_col = "alt"
    obs_id_col = "id"
    choice_col = "choice"
    mixing_id_col = "id"
    mixing_vars = ["meals", "petfr", "emipp"]
    spec, spec_names = OrderedDict(), OrderedDict()
    for col in varnames:
        df[col] = df[col].astype(float)
        spec[col] = [[1, 2, 3]]
        spec_names[col] = [col]

if profile:
    ini_ram = curr_ram()
    profiler = Profiler().start()

np.random.seed(0)
# Prints are temporarily disabled as pylogit has excessive verbosity
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()  # Disable print
model = pl.create_choice_model(data=df, alt_id_col=alt_id_col,
                               obs_id_col=obs_id_col, choice_col=choice_col,
                               specification=spec, mixing_vars=mixing_vars,
                               model_type="Mixed Logit", names=spec_names,
                               mixing_id_col=mixing_id_col)
model.fit_mle(init_vals=np.zeros(len(varnames)+len(mixing_vars)),
              num_draws=n_draws, seed=123)
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__  # Enable print

if profile:
    ellapsed, max_ram, max_gpu = profiler.stop()
    log("{:6} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {}"
        .format(n_draws, ellapsed, model.log_likelihood,
                max_ram - ini_ram, max_gpu, model.estimation_success))
    profiler.export('pylogit', dataset, n_draws, ellapsed,
                    model.log_likelihood,
                    max_ram - ini_ram, max_gpu, model.estimation_success)

if not profile:
    summ = model.summary
    names, coeff = summ.index.values, summ.parameters.values
    stderr = summ.std_err.values
    log("Variable    Estimate   Std.Err.")
    for i in range(len(names)):
        log("{:9}  {:9.5}  {:9.5}".format(names[i][:8],
                                          coeff[i], stderr[i]))
    log("Log.Lik:   {:9.2f}".format(model.log_likelihood))

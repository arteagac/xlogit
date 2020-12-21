# -*- coding: utf-8 -*-
"""
Ensures that all the dependencies were properly installed and are available
for the benchmark
"""
import os
import sys

print("Python version {}.{}".format(sys.version_info[0], sys.version_info[1]))
try:
    import cupy
    print("SUCCESS: CuPy was properly installed.")
except Exception:
    print("ERROR: CuPy was NOT properly installed")

try:
    import xlogit
    print("SUCCESS: xlogit was properly installed.")
except Exception:
    print("ERROR: xlogit was NOT properly installed")

try:
    import biogeme
    import biogeme.database as db
    import biogeme.biogeme as bio
    import biogeme.models as models
    import biogeme.messaging as msg
    from biogeme.expressions import Beta, bioDraws, log, MonteCarlo
    print("SUCCESS: biogeme was properly installed.")
except Exception:
    print("ERROR: biogeme was NOT properly installed")
    
try:
    import pylogit as pl
    print("SUCCESS: pylogit was properly installed.")
except Exception:
    print("ERROR: pylogit was NOT properly installed")

print("")
cmd = "Rscript --version"
if os.system(cmd) == 0:
    print("SUCCESS: Rscript was properly configured.")
else:
    print("ERROR: Rscript was NOT properly configured. Make sure you added"
          "the R installation folder to the Path environment variable")  

cmd = "Rscript -e 'x=suppressMessages(require(mlogit));quit(status=!x*1)'"
if os.system(cmd) == 0:
    print("SUCCESS: mlogit was properly installed.")
else:
    print("ERROR: mlogit was NOT properly installed")  
    

cmd = "Rscript -e 'x=suppressMessages(require(apollo));quit(status=!x*1)'"
if os.system(cmd) == 0:
    print("SUCCESS: apollo was properly installed.")
else:
    print("ERROR: apollo was NOT properly installed")  

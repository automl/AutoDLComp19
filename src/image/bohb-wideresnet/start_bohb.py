"""
Python script starting BOHB with the optimizer defined by the user:
    BOHB: Bayesian Optimization HyperBand
    H2BO: BOHB with 1-d KDE, min_points_in_model=10, fully_dimensional=False
    HB  : HyperBand
    RS  : Random Search
If no option is specified bohb-wideresnet.sh will be called 4 times, one per each
optimizer.
"""

import subprocess
import os
import sys

if sys.argv.__len__() != 1:
    optimizers_list = sys.argv[1:]
else:
    optimizers_list = ['BOHB', 'H2BO', 'HB', 'RS']

for optimizer in optimizers_list:
    subprocess.check_call("sbatch bohb-wideresnet.sh "+optimizer, shell=True)

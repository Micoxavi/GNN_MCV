import numpy as np
import pandas as pd
import os

basepath = "./data"
basepath_FA = os.path.join(basepath, "FA")
basepath_GM = os.path.join(basepath, "GM")
basepath_RS = os.path.join(basepath, "RS")

CT_CONTROL = -1  # Healthy volunteers without MS
CT_RRMS = 0  # Relapsing remitting MS
CT_SPMS = 1  # Secondary progressive MS
CT_PPMS = 2  # Primary progressive MS

df = pd.read_csv(os.path.join(basepath, "demographics.csv"))

target = df["mstype"].values

# Treat all patients with MS equally
target = target + 1
target[target > 1] = 1

np.unique(target, return_counts=True)
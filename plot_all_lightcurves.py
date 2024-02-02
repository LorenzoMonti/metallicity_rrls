import pandas as pd
import numpy as np
from scipy.interpolate import splrep, BSpline, make_smoothing_spline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob


dir_path = "data/rrls/catalog/"


rrl_list = list()
for file_name in glob.glob(dir_path + '*.csv'):
    x = pd.read_csv(file_name, low_memory=False)
    rrl_list.append(x)

for rrl in rrl_list:
    plt.plot(rrl["phase"], rrl["magnitude"], 'o')

ax = plt.gca()
ax.invert_yaxis()
plt.show()

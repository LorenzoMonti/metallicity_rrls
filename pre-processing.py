# -*- coding: utf-8 -*-
"""
pre-processing.py
Created on 08-11-2023 

@author: Lorenzo Monti
@email: lorenzo.monti@inaf.it

Pre-processing process in order to prepare the dataset
"""

import pandas as pd
import numpy as np
import glob
from scipy.interpolate import splrep, BSpline, make_smoothing_spline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def save_plotData(spline, source, path):
    """
    Save mag/phase plotters
    """
    listX, listY = spline[0], spline[1]
    spl, tck, tck_s = spline[2], spline[3], spline[4] 
    
    plt.plot(listX, spl(listX), '-', label='Smoothing spline')
    plt.plot(listX, BSpline(*tck)(listX), '-', label='B-spline representation s=0')
    plt.plot(listX, BSpline(*tck_s)(listX), '-', label=f'B-spline representation s={len(listX)}')

    plt.plot(listX, listY, 'o')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.legend(loc="best")
    plt.savefig(path + str(source) + ".png",  bbox_inches='tight')
    plt.close()

def compute_splines(x, y, yMean, isMean):
    """
    Create x (phase) and y (magnitude) lists and 
    compute three different splines
    """
    if isMean: listX, listY = (list(t) for t in zip(*sorted(zip(x, (y - yMean)))))
    else: listX, listY = (list(t) for t in zip(*sorted(zip(x, y))))

    spl = make_smoothing_spline(np.array(listX), np.array(listY))
    #print(f"{listX}, {spl(listX)}")
    tck = splrep(np.array(listX), np.array(listY), s=0)
    tck_s = splrep(np.array(listX), np.array(listY), s=len(x))
    return [listX, listY, spl, tck, tck_s]

def generate_augmented_data(spline, spline_points):
    """
    Generate new data based on spline
    """
    listX, listY, spl = spline[0], spline[1], spline[2] 
    # Generate new x values for interpolation
    augmented_listX = np.linspace(min(listX), max(listX), spline_points)

    # Use the spline to interpolate new y values
    augmented_listY = spl(augmented_listX)
    """print(f"{augmented_listX},\n{augmented_listY}\n")
    plt.plot(augmented_listX, augmented_listY, 'o', label='Augmented Smoothing spline', color='red')
    plt.plot(listX, spl(listX), 'o', label='Smoothing spline', color='green')
    plt.plot(listX, listY, 'o', color='grey')
    plt.show()"""
    return [augmented_listX, augmented_listY]

def create_aug_full_catolog(rrls, path, source, augm_data):
    """
    Create a dataset (catalog) using augmented 
    phase and magnitude data from spline method 
    and adding metallicity measure (FeH) from rrls 
    file as class.
    """
    # retrieve metallicity
    rrl = pd.read_csv(rrls)
    temp_dataset = rrl.loc[:,["source_id","FeH", "FeH_error"]]
    metallicity = temp_dataset.loc[temp_dataset['source_id'] == source, 'FeH'].iloc[0]
    metallicity_error = temp_dataset.loc[temp_dataset['source_id'] == source, 'FeH_error'].iloc[0]


    # write file
    with open(path + str(source) + ".csv", "w") as f:
        f.write("phase,magnitude,FeH, FeH_e\n")
        data_list = list(zip(augm_data[0], augm_data[1]))
        for data in data_list:
            f.write(f"{data[0]}, {data[1]}, {metallicity}, {metallicity_error}\n")

def create_aug_catalog(path, source, augm_data):
    """
    Create a dataset (catalog) using augmented 
    phase and magnitude data from spline method
    """
    # write file
    with open(path + str(source) + ".csv", "w") as f:
        f.write("n,time,phase,magnitude\n")
        data_list = list(zip(augm_data[0], augm_data[1]))
        for data in data_list:
            f.write(f"'', '', {data[0]}, {data[1]}\n")

def create_catalog(path, source, data, isMean):
    """
    Create a dataset (catalog) using phase  
    and magnitude data from rrls file as class.
    """
    
    phase, magn, magnMean = data[0], data[1], data[2]
    if isMean: listPhase, listMagn = (list(t) for t in zip(*sorted(zip(phase, (magn - magnMean)))))
    else: listPhase, listMagn = (list(t) for t in zip(*sorted(zip(phase, magn))))

    # write file
    with open(path + str(source) + ".csv", "w") as f:
        f.write("n,time,phase,magnitude\n")
        data_list = list(zip(listPhase, listMagn))
        for data in data_list:
            f.write(f"'', '', {data[0]}, {data[1]}\n")

def find_max_points(data_path):
    max_points = 0
    for file_name in glob.glob(data_path + '*.csv'):
        data = pd.read_csv(file_name, low_memory=False)
        if max_points < data.shape[0]: max_points = data.shape[0]
    return max_points

if __name__ == "__main__":

    dir_path = "data/rrls/raw_datasets/raw_testset_all_c/light_curves/"
    dir_rrls = "data/rrls/raw_datasets/raw_testset_all_c/rrls.csv"
    save_path = "data/rrls/plotted_dataset/"
    catalog_path = "data/rrls/catalog/"
    spline_points = find_max_points(dir_path)
    
    # dataset generation variables
    isMean = True # if you want to standardize magnitude or not
    is_plot = False # if you want to plot source splined dataset 
    is_spline = True # if you want to generate splined dataset or not
    full_catalog = False # catalog splined with phase, mean(mag), feh and feh_error (Full Catalog) 

    df_rrls = pd.read_csv(dir_rrls)
    sources_id_rrls = df_rrls.loc[:,"source_id"]

    for source in sources_id_rrls:
        try:
            print(source)
            df_lc = pd.read_csv(dir_path + str(source) + ".csv")
            source_lc = df_lc.loc[:,["fase1","mag"]]
            mag_mean = source_lc.loc[:, "mag"].mean()

            if is_spline:
                spline = compute_splines(source_lc.loc[:, "fase1"], source_lc.loc[:, "mag"], mag_mean, isMean=isMean)
                if is_plot: save_plotData(spline, source, save_path)
                augm_data = generate_augmented_data(spline, spline_points)
                if full_catalog: create_aug_full_catolog(dir_rrls, catalog_path, source, augm_data)
                else: create_aug_catalog(catalog_path, source, augm_data)
            else:
                create_catalog(catalog_path, source, [source_lc.loc[:, "fase1"], source_lc.loc[:, "mag"], mag_mean], isMean=isMean)
        except:
            print("file not found in dataset")
            continue
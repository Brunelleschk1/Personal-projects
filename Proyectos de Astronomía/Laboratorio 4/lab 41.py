import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.cluster import HDBSCAN
import astropy.units as u
# from zero_point import zpt
from tqdm import tqdm
import os
os.getcwd()

carpeta = "Proyectos de Astronomía/Laboratorio 4/"
df = pd.read_csv(carpeta + "dataset_100pc(in).csv")
df = df.dropna(subset=['ra', 'dec', 'pmra','pmdec','parallax','bp_rp','phot_g_mean_mag'])

df["dist_pc"] = 1000 / df["parallax"]
df46 = df[(df["dist_pc"] > 26) & (df["dist_pc"] < 66)]
df46 = df46[(df46["ra"]>51) & (df46["ra"] < 81)]
df46 = df46[(df46["dec"]>1) & (df46["dec"] < 31)]
print(len(df46["dec"]))

pmra46 = df46["pmra"].values
pmdec46 = df46["pmdec"].values

ra  = df["ra"].values
fidelity = df["fidelity_v2"].values

pmra  = df["pmra"].values
pmdec = df["pmdec"].values
e_ra  = df["pmra_error"].values
e_dec = df["pmdec_error"].values

"""# -------- VPD -------- #

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

titles = ["VPD Completo", "VPD Zoom Amplio", "VPD Zoom Cluster"]
xlims = [None, (-100, 300), (50, 150)]
ylims = [None, (-200, 200), (-60, 20)]

for i in range(3):

    axs[i].scatter(pmra46, pmdec46,
                   s=1,
                   alpha=0.5)

    axs[i].set_title(titles[i])

    if xlims[i] is not None:
        axs[i].set_xlim(xlims[i])

    if ylims[i] is not None:
        axs[i].set_ylim(ylims[i])

plt.tight_layout()
plt.show()"""

clustering_on = ['pmra','pmdec','parallax']

features = df46[clustering_on]

scaler = RobustScaler()
X = scaler.fit_transform(features)

hd46 = HDBSCAN(min_cluster_size=50,min_samples=10,metric='euclidean')

hd46.fit(X)

labels = hd46.labels_

df46["label_hb"] = labels

for i in np.unique(labels):
    print(i, len(df46[df46["label_hb"] == i]))


unique_labels = np.unique(labels)

plt.figure(figsize=(6,6))

for lab in unique_labels:

    cond = df46["label_hb"] == lab

    if lab == -1:
        color = "lightgray"
        label = "noise"
    else:
        color = None
        label = f"cluster {lab}"

    plt.scatter(
        df46["pmra"][cond],
        df46["pmdec"][cond],
        s=5,
        label=label
    )

plt.xlabel("pmRA")
plt.ylabel("pmDEC")
plt.title("VPD clusters")
plt.legend(markerscale=3)
plt.show()
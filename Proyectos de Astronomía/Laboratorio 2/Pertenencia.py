import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

carpeta = "Proyectos de Astronomía/Laboratorio 2/"

df = pd.read_csv(carpeta + "DatosM41Lab2")

pmra  = df["pmra"].values
pmdec = df["pmdec"].values

fig, axs = plt.subplots(1, 2, figsize=(12,6))

# -------- Panel 1: Completo --------
axs[0].scatter(pmra, pmdec, s=5, alpha=0.5)
axs[0].set_xlabel("pmRA (mas/yr)")
axs[0].set_ylabel("pmDEC (mas/yr)")
axs[0].set_title("Vector Point Diagram - Completo")
axs[0].grid(alpha=0.3)

# -------- Panel 2: Acotado --------
axs[1].scatter(pmra, pmdec, s=5, alpha=0.5)
axs[1].set_xlim(-20, 20)
axs[1].set_ylim(-30, 30)
axs[1].set_xlabel("pmRA (mas/yr)")
axs[1].set_ylabel("pmDEC (mas/yr)")
axs[1].set_title("Vector Point Diagram - Zoom")
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(carpeta + "DiagramasVP/DiagramaVPFully3020")
plt.close()

from scipy.optimize import minimize

def neg_log_likelihood(params, pmra, pmdec):

    mu_ra_c, mu_dec_c, sigma_c, \
    mu_ra_f, mu_dec_f, sigma_f, \
    f_c = params

    # cluster gaussian
    phi_c = (1/(2*np.pi*sigma_c**2)) * np.exp(
        -((pmra-mu_ra_c)**2 + (pmdec-mu_dec_c)**2)/(2*sigma_c**2)
    )

    # field gaussian
    phi_f = (1/(2*np.pi*sigma_f**2)) * np.exp(
        -((pmra-mu_ra_f)**2 + (pmdec-mu_dec_f)**2)/(2*sigma_f**2)
    )

    phi = f_c*phi_c + (1-f_c)*phi_f

    return -np.sum(np.log(phi + 1e-10))
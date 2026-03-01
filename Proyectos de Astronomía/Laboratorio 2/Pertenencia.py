import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
carpeta = "Proyectos de Astronomía/Laboratorio 2/"

df = pd.read_csv(carpeta + "DatosM41Lab2")

pmra  = df["pmra"].values
pmdec = df["pmdec"].values

# ============================================================
# MODELO COMPLETO CON BOUNDS Y L-BFGS-B
# ============================================================

def ajuste_membresia_vpd_completo(pmra, pmdec, initial_guess):

    pmra = np.asarray(pmra)
    pmdec = np.asarray(pmdec)

    mask = np.isfinite(pmra) & np.isfinite(pmdec)
    pmra = pmra[mask]
    pmdec = pmdec[mask]

    def neg_log_likelihood(params):

        mu_xc, mu_yc, sigma_c, \
        mu_xf, mu_yf, sigma_xf, sigma_yf, \
        rho, n_f = params

        # Cluster
        phi_c = (1/(2*np.pi*sigma_c**2)) * np.exp(
            -((pmra-mu_xc)**2 + (pmdec-mu_yc)**2)/(2*sigma_c**2)
        )

        # Field correlacionado
        dx = pmra - mu_xf
        dy = pmdec - mu_yf

        norm_f = 1/(2*np.pi*sigma_xf*sigma_yf*np.sqrt(1-rho**2))

        exponent_f = (
            (dx**2)/(sigma_xf**2)
            + (dy**2)/(sigma_yf**2)
            - (2*rho*dx*dy)/(sigma_xf*sigma_yf)
        ) / (2*(1-rho**2))

        phi_f = norm_f * np.exp(-exponent_f)

        phi = (1-n_f)*phi_c + n_f*phi_f

        return -np.sum(np.log(phi + 1e-15))


    # -----------------------------
    # Bounds físicos
    # -----------------------------
    bounds = [
        (-20, 20),     # mu_xc
        (-30, 30),     # mu_yc
        (1e-3, 5),     # sigma_c
        (-20, 20),     # mu_xf
        (-30, 30),     # mu_yf
        (1e-3, 20),    # sigma_xf
        (1e-3, 20),    # sigma_yf
        (-0.99, 0.99), # rho
        (1e-4, 0.9999) # n_f
    ]

    result = minimize(
        neg_log_likelihood,
        initial_guess,
        method="L-BFGS-B",
        bounds=bounds
    )

    mu_xc, mu_yc, sigma_c, \
    mu_xf, mu_yf, sigma_xf, sigma_yf, \
    rho, n_f = result.x

    # Recalcular probabilidades
    phi_c = (1/(2*np.pi*sigma_c**2)) * np.exp(
        -((pmra-mu_xc)**2 + (pmdec-mu_yc)**2)/(2*sigma_c**2)
    )

    dx = pmra - mu_xf
    dy = pmdec - mu_yf

    norm_f = 1/(2*np.pi*sigma_xf*sigma_yf*np.sqrt(1-rho**2))
    exponent_f = (
        (dx**2)/(sigma_xf**2)
        + (dy**2)/(sigma_yf**2)
        - (2*rho*dx*dy)/(sigma_xf*sigma_yf)
    ) / (2*(1-rho**2))

    phi_f = norm_f * np.exp(-exponent_f)

    P_cluster = ((1-n_f)*phi_c) / ((1-n_f)*phi_c + n_f*phi_f)

    return result.x, P_cluster, result

# ============================================================
# PARÁMETROS INICIALES (LOS TUYOS)
# ============================================================

initial_guess = [
    -4.3668,   # mu_x cluster
    -1.3399,   # mu_y cluster
     0.2633,   # sigma cluster
    -0.7410,   # mu_x field
     0.6275,   # mu_y field
     5.1546,   # sigma_x field
     7.3140,   # sigma_y field
    -0.2815,   # rho
     0.9455    # n_f
]


# ============================================================
# EJECUTAR AJUSTE
# ============================================================

opt, P, result = ajuste_membresia_vpd_completo(pmra, pmdec, initial_guess)

mu_xc, mu_yc, sigma_c, \
mu_xf, mu_yf, sigma_xf, sigma_yf, \
rho, n_f = opt

print("\n=========== RESULTADO FINAL ===========")
print("mu_x_cluster   =", opt[0])
print("mu_y_cluster   =", opt[1])
print("sigma_cluster  =", opt[2])
print("mu_x_field     =", opt[3])
print("mu_y_field     =", opt[4])
print("sigma_x_field  =", opt[5])
print("sigma_y_field  =", opt[6])
print("rho_field      =", opt[7])
print("n_field        =", opt[8])
print("n_cluster      =", 1 - opt[8])
print("Convergencia   =", result.success)
print("=======================================")


# ============================================================
# VPD CON LÍMITES PEDIDOS
# ============================================================

# ============================================================
# VPD CON PROBABILIDADES (3 ZOOMS)
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(18,6))

# -------- Panel 1: Completo --------
axs[0].scatter(pmra, pmdec, c=P, s=5, cmap='viridis')
axs[0].set_title("VPD Completo (con membresía)")
axs[0].set_xlabel("pmRA (mas/yr)")
axs[0].set_ylabel("pmDEC (mas/yr)")
axs[0].grid(alpha=0.3)

# -------- Panel 2: Zoom amplio --------
axs[1].scatter(pmra, pmdec, c=P, s=5, cmap='viridis')
axs[1].set_xlim(-20, 20)
axs[1].set_ylim(-30, 30)
axs[1].set_title("VPD Zoom (-20,20 / -30,30)")
axs[1].set_xlabel("pmRA (mas/yr)")
axs[1].set_ylabel("pmDEC (mas/yr)")
axs[1].grid(alpha=0.3)

# -------- Panel 3: Zoom pequeño solicitado --------
axs[2].scatter(pmra, pmdec, c=P, s=5, cmap='viridis')
axs[2].set_xlim(-7, -2)
axs[2].set_ylim(-5, 1)
axs[2].set_title("VPD Zoom (-7,-2 / -5,1)")
axs[2].set_xlabel("pmRA (mas/yr)")
axs[2].set_ylabel("pmDEC (mas/yr)")
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(carpeta + "DiagramasVP/DiagramaVPTripleConPertenencia")
plt.close()

# ============================================================
# COMPARACIÓN: VPD SIN CLUSTER RESALTADO
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(18,6))

# -------- Panel 1: Completo --------
axs[0].scatter(pmra, pmdec, s=5, alpha=0.5)
axs[0].set_title("VPD Completo (sin membresía)")
axs[0].set_xlabel("pmRA (mas/yr)")
axs[0].set_ylabel("pmDEC (mas/yr)")
axs[0].grid(alpha=0.3)

# -------- Panel 2: Zoom amplio --------
axs[1].scatter(pmra, pmdec, s=5, alpha=0.5)
axs[1].set_xlim(-20, 20)
axs[1].set_ylim(-30, 30)
axs[1].set_title("VPD Zoom (-20,20 / -30,30)")
axs[1].set_xlabel("pmRA (mas/yr)")
axs[1].set_ylabel("pmDEC (mas/yr)")
axs[1].grid(alpha=0.3)

# -------- Panel 3: Zoom pequeño solicitado --------
axs[2].scatter(pmra, pmdec, s=5, alpha=0.5)
axs[2].set_xlim(-7, -2)
axs[2].set_ylim(-5, 1)
axs[2].set_title("VPD Zoom (-7,-2 / -5,1)")
axs[2].set_xlabel("pmRA (mas/yr)")
axs[2].set_ylabel("pmDEC (mas/yr)")
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(carpeta + "DiagramasVP/DiagramaVPTripleSinPertenencia")
plt.close()

# ============================================================
# CONTEO DE ESTRELLAS
# ============================================================

N_total = len(P)

threshold = 0.7
mask_cluster = P > threshold

N_cluster_coloreadas = np.sum(mask_cluster)
N_field = N_total - N_cluster_coloreadas

print("\n=========== CONTEOS ===========")
print("Total de estrellas           =", N_total)
print(f"Estrellas P > {threshold}     =", N_cluster_coloreadas)
print("Estrellas clasificadas campo =", N_field)
print("Fracción cluster observada   =", N_cluster_coloreadas / N_total)
print("Fracción field observada     =", N_field / N_total)
print("================================")

# ============================================================
# HISTOGRAMA DE PROBABILIDADES
# ============================================================

plt.figure(figsize=(6,5))
plt.hist(P, bins=10)
plt.xlabel("Probabilidad de pertenencia al cluster")
plt.ylabel("Número de estrellas")
plt.title("Distribución de probabilidades de membresía")
plt.grid(alpha=0.3)
plt.show()

# ============================================================
# ESTUDIO DEL THRESHOLD
# ============================================================

thresholds = np.linspace(0, 1, 200)
N_above = [np.sum(P > t) for t in thresholds]

plt.figure(figsize=(6,5))
plt.plot(thresholds, N_above)
plt.xlabel("Threshold de probabilidad")
plt.ylabel("Número de estrellas con P > threshold")
plt.title("Dependencia del número de miembros con threshold")
plt.grid(alpha=0.3)
plt.show()

# ============================================================
# DIAGRAMA COLOR-MAGNITUD (CMD)
# ============================================================

# Extraer columnas fotométricas
Gmag = df["phot_g_mean_mag"].values
color = df["bp_rp"].values

# Aplicar misma máscara de valores finitos
mask_cmd = np.isfinite(Gmag) & np.isfinite(color)
Gmag = Gmag[mask_cmd]
color = color[mask_cmd]
P_cmd = P[mask_cmd]

threshold = 0.7
mask_cluster = P_cmd > threshold
mask_field = P_cmd <= threshold

plt.figure(figsize=(6,8))

# Campo primero (fondo)
plt.scatter(color[mask_field], Gmag[mask_field],
            s=5, alpha=0.3, label="Campo")

# Cluster resaltado
plt.scatter(color[mask_cluster], Gmag[mask_cluster],
            s=8, label="Cluster")

plt.gca().invert_yaxis()  # magnitud crece hacia abajo
plt.xlabel("BP - RP")
plt.ylabel("G")
plt.title("Diagrama Color-Magnitud (Cluster resaltado)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

carpeta = "Proyectos de Astronomía/Laboratorio 2/"

# Carga de datos
df = pd.read_csv(carpeta + "DatosM41Lab2")

# Extraer valores y sus errores
pmra  = df["pmra"].values
pmdec = df["pmdec"].values
e_ra  = df["pmra_error"].values
e_dec = df["pmdec_error"].values
Gmag  = df["phot_g_mean_mag"].values
color = df["bp_rp"].values

# Máscara única para asegurar que todos los vectores tengan el mismo tamaño
mask = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(e_ra) & \
       np.isfinite(e_dec) & np.isfinite(Gmag) & np.isfinite(color)

pmra, pmdec = pmra[mask], pmdec[mask]
e_ra, e_dec = e_ra[mask], e_dec[mask]
Gmag, color = Gmag[mask], color[mask]

# ============================================================
# MODELO COMPLETO CON BOUNDS Y L-BFGS-B
# ============================================================

def ajuste_membresia_vpd_completo(pmra, pmdec, e_ra, e_dec, initial_guess):
    
    def neg_log_likelihood(params):
        mu_xc, mu_yc, sigma_c, \
        mu_xf, mu_yf, sigma_xf, sigma_yf, \
        rho, n_f = params

        # --- CLUSTER CON ERROR ---
        sig_c_tot_x2 = sigma_c**2 + e_ra**2
        sig_c_tot_y2 = sigma_c**2 + e_dec**2
        
        phi_c = (1 / (2 * np.pi * np.sqrt(sig_c_tot_x2 * sig_c_tot_y2))) * np.exp(
            -0.5 * (((pmra - mu_xc)**2 / sig_c_tot_x2) + ((pmdec - mu_yc)**2 / sig_c_tot_y2))
        )

        # --- CAMPO CON ERROR ---
        dx, dy = pmra - mu_xf, pmdec - mu_yf
        s_xf_tot2, s_yf_tot2 = sigma_xf**2 + e_ra**2, sigma_yf**2 + e_dec**2
        s_xf_tot, s_yf_tot = np.sqrt(s_xf_tot2), np.sqrt(s_yf_tot2)

        norm_f = 1 / (2 * np.pi * s_xf_tot * s_yf_tot * np.sqrt(1 - rho**2))
        exponent_f = (
            (dx**2 / s_xf_tot2) + (dy**2 / s_yf_tot2) - 
            (2 * rho * dx * dy / (s_xf_tot * s_yf_tot))
        ) / (2)  # ⚠️ CORREGIR: dividir por 2*(1 - rho**2)

        phi_f = norm_f * np.exp(-exponent_f)
        
        # Mezcla Bayesiana
        phi = (1 - n_f) * phi_c + n_f * phi_f
        return -np.sum(np.log(phi + 1e-15))

    # Bounds físicos
    bounds = [
        (-20, 20), (-30, 30), (1e-3, 5),     # Cluster
        (-20, 20), (-30, 30), (1e-3, 20),    # Field pos/sigma
        (1e-3, 20), (-0.99, 0.99),           # Sigma_y_f, rho
        (1e-4, 0.9999)                       # n_f
    ]

    result = minimize(neg_log_likelihood, initial_guess, method="L-BFGS-B", bounds=bounds)
    
    # --- CÁLCULO FINAL DE PROBABILIDADES (BAYES) ---
    # Usamos los parámetros optimizados para calcular la probabilidad suave
    p = result.x
    n_f_opt = p[8]
    
    # Volvemos a calcular phi_c y phi_f con los errores para P_cluster
    s_c_x2, s_c_y2 = p[2]**2 + e_ra**2, p[2]**2 + e_dec**2
    phi_c_final = (1 / (2 * np.pi * np.sqrt(s_c_x2 * s_c_y2))) * np.exp(
        -0.5 * (((pmra - p[0])**2 / s_c_x2) + ((pmdec - p[1])**2 / s_c_y2))
    )
    
    dx_f, dy_f = pmra - p[3], pmdec - p[4]
    sx_f2, sy_f2 = p[5]**2 + e_ra**2, p[6]**2 + e_dec**2
    sxf, syf = np.sqrt(sx_f2), np.sqrt(sy_f2)
    
    norm_f = 1 / (2 * np.pi * sxf * syf * np.sqrt(1 - p[7]**2))
    exp_f = ((dx_f**2/sx_f2) + (dy_f**2/sy_f2) - (2*p[7]*dx_f*dy_f/(sxf*syf))) / (2) # ⚠️ FALTA dividir por (1 - p[7]**2)
    phi_f_final = norm_f * np.exp(-exp_f)

    # Fórmula Bayesiana: P = (Nc * Phic) / (Nc * Phic + Nf * Phif)
    P_cluster = ((1 - n_f_opt) * phi_c_final) / ((1 - n_f_opt) * phi_c_final + n_f_opt * phi_f_final)

    return result.x, P_cluster, result

# ============================================================
# PARÁMETROS INICIALES (LOS TUYOS)
# ============================================================

initial_guess = [-4.36, -1.33, 0.26, -0.74, 0.62, 5.15, 7.31, -0.28, 0.95]

opt, P, result = ajuste_membresia_vpd_completo(pmra, pmdec, e_ra, e_dec, initial_guess)

print("\n=========== RESULTADO FINAL ===========")
print(f"n_cluster (modelo) = {1 - opt[8]:.4f}")
print(f"Convergencia       = {result.success}")
print("=======================================")

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
# CONTEO DE ESTRELLAS
# ============================================================

IntenoN = len(color)


N_total = len(P)

# ----- Conteo duro (threshold) -----
threshold = 0.7
mask_cluster = P > threshold
N_cluster_hard = np.sum(mask_cluster)
N_field_hard = N_total - N_cluster_hard
# ----- Conteo suave (Bayesiano real) -----
# La suma de probabilidades es el estimador esperado del número de miembros
N_cluster_soft = np.sum(P)
N_field_soft = N_total - N_cluster_soft

# ----- Conteo esperado del modelo -----
# Según el parámetro ajustado
n_f_model = opt[8]
N_cluster_model = (1 - n_f_model) * N_total
N_field_model = n_f_model * N_total

print("\n=========== CONTEO DE ESTRELLAS ===========")
print("Total estrella brunito          =", IntenoN)
print("Total estrellas                 =", N_total)
print("\n--- Conteo duro (P > 0.7) ---")
print("Cluster (hard)                  =", N_cluster_hard)
print("Field (hard)                    =", N_field_hard)

print("\n--- Conteo suave (Σ P_i) ---")
print("Cluster (soft esperado)         =", N_cluster_soft)
print("Field (soft esperado)           =", N_field_soft)

print("\n--- Según parámetro n_f ---")
print("Cluster (modelo)                =", N_cluster_model)
print("Field (modelo)                  =", N_field_model)
print("============================================")

# ============================================================
# VPD CON LÍMITES PEDIDOS
# ============================================================

# ============================================================
# VPD CON PROBABILIDADES (3 ZOOMS)
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
titles = ["VPD Completo", "VPD Zoom Amplio", "VPD Zoom Cluster"]
xlims = [None, (-20, 20), (-7, -2)]
ylims = [None, (-30, 30), (-5, 1)]

for i in range(3):
    sc = axs[i].scatter(pmra, pmdec, c=P, s=5, cmap='viridis', alpha=0.6)
    axs[i].set_title(titles[i])
    if xlims[i]: axs[i].set_xlim(xlims[i])
    if ylims[i]: axs[i].set_ylim(ylims[i])
    axs[i].grid(alpha=0.3)

plt.colorbar(sc, ax=axs[2], label='Probabilidad Membresía')
plt.tight_layout()
plt.savefig(carpeta + "CosasErrNoNorm/VPDColoreado")
plt.close()

# Histograma de Probabilidades
plt.figure(figsize=(6, 4))
plt.hist(P, bins=10, color='skyblue', edgecolor='black')
plt.xlabel("Probabilidad de pertenencia")
plt.ylabel("N° Estrellas")
plt.title("Distribución de Probabilidades (Bayesiana)")
plt.yscale('log') # Escala log para ver mejor el cúmulo frente al campo
plt.savefig(carpeta + "CosasErrNoNorm/ProbPertU")
plt.close()

# CMD
plt.figure(figsize=(6, 8))
mask_c = P > 0.7
plt.scatter(color[~mask_c], Gmag[~mask_c], s=2, alpha=0.2, c='gray', label='Campo')
plt.scatter(color[mask_c], Gmag[mask_c], s=10, c=P[mask_c], cmap='viridis', label='Cluster')
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("G")
plt.title("CMD con Probabilidades")
plt.legend()
plt.savefig(carpeta + "CosasErrNoNorm/CMDPintadoErr")
plt.close()
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from skimage.feature import peak_local_max
import astroalign as aa

ruta = "Proyectos de Astronomía/Laboratorio 1/M41.fits"
carpeta1 = "Proyectos de Astronomía/Laboratorio 1/FotosNormales/"
carpeta2 = "Proyectos de Astronomía/Laboratorio 1/Centroides/"
carpeta3 = "Proyectos de Astronomía/Laboratorio 1/CentroidesFiltrados/"
carpeta = "Proyectos de Astronomía/Laboratorio 1/"

Azul = fits.getdata(ruta, ext=0)
Rojo = fits.getdata(ruta, ext=1)
Verde = fits.getdata(ruta, ext=2)

Azul  = np.array(Azul,  dtype=np.float64)
Rojo  = np.array(Rojo,  dtype=np.float64)
Verde = np.array(Verde, dtype=np.float64)

Verdea, footprint_v = aa.register(Verde, Rojo)
Azula, footprint_r  = aa.register(Azul, Rojo)

def normalizar(img):
    p1, p99 = np.percentile(img[np.isfinite(img)], (1, 99))
    img_clip = np.clip(img, p1, p99)
    return (img_clip - p1) / (p99 - p1)

rgb = np.zeros((Rojo.shape[0], Rojo.shape[1], 3))

rgb[:,:,0] = normalizar(Rojo)
rgb[:,:,1] = normalizar(Verdea)
rgb[:,:,2] = normalizar(Azula)

plt.figure(figsize=(8,8))
plt.imshow(rgb)
plt.title("RGB alineado sobre rojo (percentil 1–99)")
plt.savefig(carpeta + "Alineamiento RGB Sobre Rojo",dpi=300, bbox_inches="tight")
plt.close()

def detectar_centroides(imagen, nombre_filtro, carpeta,min_distance=20,threshold_factor=15,box_size=7):

    mean, median, std = sigma_clipped_stats(imagen, sigma=1.5)
    imagen2 = imagen - median

    plt.figure(figsize=(10,10))
    plt.imshow(imagen2, cmap='magma',vmin=median-std,vmax=median+5*std)
    plt.title(f"{nombre_filtro} fondo restado")
    plt.savefig(carpeta1 + f"{nombre_filtro}_FondoRestado.png",dpi=300, bbox_inches="tight")
    plt.close()

    coords = peak_local_max(imagen2,min_distance=min_distance,threshold_abs=threshold_factor*std)

    x_peaks = coords[:,1]
    y_peaks = coords[:,0]
    x_centroids = []
    y_centroids = []
    half_box = box_size // 2
    ny, nx = imagen.shape

    for x, y in zip(x_peaks, y_peaks):

        if (x-half_box < 0 or x+half_box >= nx or
            y-half_box < 0 or y+half_box >= ny):
            continue

        sub = imagen2[y-half_box:y+half_box+1,
                      x-half_box:x+half_box+1]

        if np.argmax(sub) != (box_size**2)//2:
            continue

        if np.max(sub) < threshold_factor*std:
            continue

        x_c, y_c = centroid_quadratic(sub)

        x_centroids.append(x - half_box + x_c)
        y_centroids.append(y - half_box + y_c)

    x_centroids = np.array(x_centroids)
    y_centroids = np.array(y_centroids)
    mask = np.isfinite(x_centroids) & np.isfinite(y_centroids)
    x_centroids = x_centroids[mask]
    y_centroids = y_centroids[mask]

    plt.figure(figsize=(10,10))
    plt.imshow(imagen, cmap='magma',vmin=median-std,vmax=median+5*std)
    plt.scatter(x_centroids, y_centroids,s=40, edgecolor='red', facecolor='none')
    plt.title(f"Centroides - {nombre_filtro}")
    plt.savefig(carpeta2 + f"Centroides_{nombre_filtro}.png",dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.imshow(imagen2, cmap='magma',vmin=median-std,vmax=median + 5*std)
    plt.scatter(x_centroids, y_centroids,s=40, edgecolor='cyan', facecolor='none')
    plt.title(f"Centroides - {nombre_filtro} (Fondo Restado)")
    plt.savefig(carpeta3 + f"Centroides_{nombre_filtro}_FondoRestado.png",dpi=300, bbox_inches="tight")
    plt.close()

    return x_centroids, y_centroids

x_verde, y_verde = detectar_centroides(Verdea, "Verde", carpeta)
x_azul,  y_azul  = detectar_centroides(Azula,  "Azul",  carpeta)
x_rojo,  y_rojo  = detectar_centroides(Rojo,  "Rojo",  carpeta)

print("\nResumen total:")
print(f"Verde: {len(x_verde)}")
print(f"Azul:  {len(x_azul)}")
print(f"Rojo:  {len(x_rojo)}")

def fotometria_apertura(imagen, x, y,r_ap=5,r_in=7,r_out=10):
    posiciones = np.transpose((x, y))
    # Apertura circular
    aperturas = CircularAperture(posiciones, r=r_ap)
    # Anillo para fondo local
    anillos = CircularAnnulus(posiciones, r_in=r_in, r_out=r_out)
    # Fotometría
    tabla = aperture_photometry(imagen, aperturas)
    tabla_anillo = aperture_photometry(imagen, anillos)
    # Área
    area_ap = aperturas.area
    area_an = anillos.area
    # Fondo medio por píxel
    fondo_medio = tabla_anillo['aperture_sum'] / area_an
    # Flujo corregido
    flujo = tabla['aperture_sum'] - fondo_medio * area_ap
    return np.array(flujo)

r_ap  = 5      # radio de apertura
r_in  = 7      # inicio anillo
r_out = 10     # fin anillo

x_ref = x_azul
y_ref = y_azul

flujo_verde = fotometria_apertura(Verde, x_ref, y_ref)
flujo_azul  = fotometria_apertura(Azul,  x_ref, y_ref)
flujo_rojo  = fotometria_apertura(Rojo,  x_ref, y_ref)

mask = (flujo_verde > 0) & (flujo_azul > 0) & (flujo_rojo > 0)

flujo_verde = flujo_verde[mask]
flujo_azul  = flujo_azul[mask]
flujo_rojo  = flujo_rojo[mask]
print(len(flujo_azul), len(flujo_verde), len(flujo_rojo))

x_ref = x_ref[mask]
y_ref = y_ref[mask]

mag_verde = -2.5 * np.log10(flujo_verde)
mag_azul  = -2.5 * np.log10(flujo_azul)
mag_rojo  = -2.5 * np.log10(flujo_rojo)
color_ar = mag_azul - mag_rojo
color_av = mag_azul - mag_verde
color_vr = mag_verde - mag_rojo

fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True)

axs[0].scatter(color_ar, mag_verde, s=10)
axs[0].set_xlabel("Azul - Rojo")
axs[0].set_ylabel("Mag Azul")

axs[1].scatter(color_av, mag_verde, s=10)
axs[1].set_xlabel("Azul - Verde")

axs[2].scatter(color_vr, mag_verde, s=10)
axs[2].set_xlabel("Verde - Rojo")

for ax in axs:
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

plt.suptitle("Diagramas Color–Magnitud")
plt.tight_layout()
plt.savefig(carpeta + "CMD_TresPaneles.png", dpi=300)
plt.show()
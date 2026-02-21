from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_quadratic
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
ruta = "Proyectos de Astronomía/Laboratorio 1/M41.fits"
carpeta1 = "Proyectos de Astronomía/Laboratorio 1/FotosNormales/"
carpeta2 = "Proyectos de Astronomía/Laboratorio 1/Centroides/"
carpeta = "Proyectos de Astronomía/Laboratorio 1/"
Azul = fits.getdata(ruta, ext=0)
Rojo = fits.getdata(ruta, ext=1)
Verde = fits.getdata(ruta, ext=2)

def detectar_centroides(imagen, nombre_filtro, carpeta,
                        min_distance=15,
                        threshold_factor=10,
                        box_size=7):

    mean, median, std = sigma_clipped_stats(imagen, sigma=1.5)
    imagen2 = imagen - median

    # Guardar fondo restado
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

    # Guardar imagen con centroides
    plt.figure(figsize=(10,10))
    plt.imshow(imagen, cmap='magma',vmin=median-std,vmax=median+5*std)
    plt.scatter(x_centroids, y_centroids,s=40, edgecolor='red', facecolor='none')
    plt.title(f"Centroides - {nombre_filtro}")
    plt.savefig(carpeta2 + f"Centroides_{nombre_filtro}.png",dpi=300, bbox_inches="tight")
    plt.close()

    return x_centroids, y_centroids

x_verde, y_verde = detectar_centroides(Verde, "Verde", carpeta)
x_azul,  y_azul  = detectar_centroides(Azul,  "Azul",  carpeta)
x_rojo,  y_rojo  = detectar_centroides(Rojo,  "Rojo",  carpeta)
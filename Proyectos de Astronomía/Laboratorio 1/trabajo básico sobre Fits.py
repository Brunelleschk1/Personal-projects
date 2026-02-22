import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
ruta = "Proyectos de Astronomía/Laboratorio 1/M41.fits"
data = fits.getdata(ruta)

Azul = fits.getdata(ruta, ext=0) 
Rojo = fits.getdata(ruta, ext=1) 
Verde = fits.getdata(ruta, ext=2)
# Calcular percentiles para eliminar extremos
vmin, vmax = np.percentile(data, [1, 99])

# Filtrar datos en ese rango
data_recortado = data[(data >= vmin) & (data <= vmax)]

mean_azul, median_azul, std_azul=sigma_clipped_stats(Azul, sigma = 2)
sigmaA = 3*std_azul 
mean_v, median_v, std_v=sigma_clipped_stats(Verde, sigma = 2)
sigmaV = std_v*3
mean_r, median_r, std_r=sigma_clipped_stats(Rojo, sigma = 2)
sigmaR = 3*std_r
Azul_Suave = gaussian_filter(Azul, sigma=sigmaA)
Verde_Suave = gaussian_filter(Verde, sigma=sigmaV)
Rojo_Suave = gaussian_filter(Rojo, sigma=sigmaR)

ImagenAzul1 = plt.imshow(Azul_Suave, cmap='magma',vmin=median_azul-std_azul, vmax=median_azul+5*std_azul)
plt.title("M41_Azul_gauss")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/Gauss/Azul Gauss",dpi=300,bbox_inches="tight")
plt.close()
ImagenV1 = plt.imshow(Verde_Suave, cmap='magma',vmin=median_v-std_v, vmax=median_v+5*std_v)
plt.title("M41_Verde_gauss")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/Gauss/Verde Gauss",dpi=300,bbox_inches="tight")
plt.close()
ImagenR1 = plt.imshow(Rojo_Suave, cmap='magma',vmin=median_r-std_r, vmax=median_r+5*std_r)
plt.title("M41_rojo_gauss")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/Gauss/Rojo Gauss",dpi=300,bbox_inches="tight")
plt.close()

plt.figure(figsize=(15, 5))
plt.hist(data.ravel(), bins=np.linspace(0, 2000, 500), log=True)
plt.xlabel("Valor del pixel")
plt.ylabel("Frecuencia")
plt.title("Histograma recortado (1–99 percentil)")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/Histograma.png")
plt.close()

plt.imshow(data, cmap="magma", norm='log')
plt.colorbar(label="Intensidad (escala log)")
plt.title("M41")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/M41.png",dpi=300,bbox_inches="tight")
plt.close()
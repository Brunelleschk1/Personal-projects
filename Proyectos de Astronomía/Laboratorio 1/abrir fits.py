from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_quadratic
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
ruta = "Proyectos de Astronomía/Laboratorio 1/M41.fits"

Azul = fits.getdata(ruta, ext=0)
Rojo = fits.getdata(ruta, ext=1)
Verde = fits.getdata(ruta, ext=2)

#Se propone una estimación del fondo para eliminarlo 
mean_azul, median_azul, std_azul=sigma_clipped_stats(Azul, sigma = 3.0)
Azul2 = Azul- median_azul

"""sigma = 2.355*std_azul
Azul_Suave = gaussian_filter(Azul, sigma=sigma)"""
#La imagen de azul suave no tiene casi nada, no parece que vaya a funcionar. Revisar histograma
ImagenAzul1 = plt.imshow(Azul2, cmap='magma',
           vmin=median_azul-std_azul,
           vmax=median_azul+5*std_azul)


coordenadas = peak_local_max(
    Azul2,
    min_distance=30,
    threshold_abs=5*std_azul
)

y_peaks = coordenadas[:, 0]
x_peaks = coordenadas[:, 1]
plt.figure(figsize=(6,6))
plt.imshow(Azul2, cmap='magma') #¿Como poner la imagen de arriba y superponerla a los picos?
plt.scatter(x_peaks, y_peaks, s=30, edgecolor='red', facecolor='none')
plt.title("Picos detectados - Blue")
plt.show()
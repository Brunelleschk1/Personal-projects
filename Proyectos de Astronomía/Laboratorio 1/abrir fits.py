from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
ruta = "Proyectos de Astronomía/Laboratorio 1/M41.fits"
data = fits.getdata(ruta)

print(fits.getval(ruta, "Filter", ext=0))
print(fits.getval(ruta, "Filter", ext=1))
print(fits.getval(ruta, "Filter", ext=2))

# Calcular percentiles para eliminar extremos
vmin, vmax = np.percentile(data, [1, 99])

# Filtrar datos en ese rango
data_recortado = data[(data >= vmin) & (data <= vmax)]

plt.figure(figsize=(15, 5))
plt.hist(data_recortado.ravel(), bins=500, log=True)
plt.xlabel("Valor del pixel")
plt.ylabel("Frecuencia")
plt.title("Histograma recortado (1–99 percentil)")
plt.savefig("Proyectos de Astronomía/Laboratorio 1/Histograma.png")
plt.show()

plt.imshow(data, cmap="magma", norm='log')
plt.colorbar(label="Intensidad (escala log)")
plt.title("M41")
plt.savefig(
    "Proyectos de Astronomía/Laboratorio 1/M41.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
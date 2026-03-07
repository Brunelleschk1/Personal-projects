import matplotlib.pyplot as plt
import pandas as pd

carpeta = "Proyectos de Astronomía/Laboratorio 2/Lab 2 Otro csv/"

df1 = pd.read_csv(carpeta + "DatosM41Lab2")
df2 = pd.read_csv(carpeta + "M41_pm_lab2.csv")

print("Viejo:", len(df1))
print("Nuevo:", len(df2))

print(df1.describe())
print(df2.describe())

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(df1["pmra"], bins=50, alpha=0.5, label="viejo")
plt.hist(df2["pmra"], bins=50, alpha=0.5, label="nuevo")
plt.legend()
plt.title("pmRA")

plt.subplot(1,2,2)
plt.hist(df1["pmdec"], bins=50, alpha=0.5, label="viejo")
plt.hist(df2["pmdec"], bins=50, alpha=0.5, label="nuevo")
plt.legend()
plt.title("pmDEC")

plt.show()

plt.figure(figsize=(6,6))

plt.scatter(df1.pmra, df1.pmdec, s=2, alpha=0.3, label="viejo")
plt.scatter(df2.pmra, df2.pmdec, s=2, alpha=0.3, label="nuevo")

plt.legend()
plt.xlabel("pmRA")
plt.ylabel("pmDEC")
plt.title("Comparación VPD")

solo_viejo = df1[~df1.source_id.isin(df2.source_id)]
solo_nuevo = df2[~df2.source_id.isin(df1.source_id)]

print("Solo en viejo:", len(solo_viejo))
print("Solo en nuevo:", len(solo_nuevo))

plt.figure(figsize=(6,6))

plt.scatter(df1.ra, df1.dec, s=2, label="viejo")
plt.scatter(df2.ra, df2.dec, s=2, label="nuevo")

plt.legend()
plt.xlabel("RA")
plt.ylabel("DEC")

from astropy.coordinates import SkyCoord
import astropy.units as u

c1 = SkyCoord(df1.ra, df1.dec, unit="deg")
c2 = SkyCoord(df2.ra, df2.dec, unit="deg")

idx, d2d, _ = c1.match_to_catalog_sky(c2)

match = d2d < 1*u.arcsec

print("Matches:", match.sum())
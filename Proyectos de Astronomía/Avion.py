import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parámetros físicos
# ----------------------------
Q = 5e4            # lúmenes del avión
F_vega = 2.56e-6   # lm/m^2
h = np.concatenate((np.linspace(0, 12000, 678), np.full(1200-678,12000)))     # altura del avión, empieza a descender a 30 km de la pista, antes de eso h es cte

# Trayectoria horizontal
x = np.linspace(4000, 50e3, 1200)  # desde 4000m hasta 40 km, un avión empieza su descenso unos 30km antes de la pista, la pista es de hasta 4 km

# ----------------------------
# Geometría
# ----------------------------
d = np.sqrt(x**2 + h**2)       # distancia real
cosZ = h / d                  # coseno del ángulo cenital

# ----------------------------
# Magnitud sin corrección
# ----------------------------
F = Q / (4 * np.pi * d**2)
m = -2.5 * np.log10(F / F_vega)

# ----------------------------
# Magnitud corregida
# ----------------------------
M = np.zeros(1200, dtype=float)

for i in range(len(cosZ)):
    if cosZ[i]<=0.2:
        M[i] = m[i] +1/1000*cosZ[i]
    else:
        M[i] = m[i] + 1/cosZ[i]
# ----------------------------
# Gráfica
# ----------------------------
plt.figure()
plt.plot(x / 1000, M, label="M(x) = m + 1/cos(Z)")
plt.plot(x/1000, m, label= "M=m" )
plt.axhline(6, linestyle="--", label="Límite ojo humano")

plt.gca().invert_yaxis()
plt.xlabel("Distancia horizontal a la pista (km)")
plt.ylabel("Magnitud aparente corregida")
plt.title("Magnitud aparente de un avión en aproximación")
plt.legend()
plt.grid(True)

plt.show()

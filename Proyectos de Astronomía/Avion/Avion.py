import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
Q = 6e4            # lúmenes del avión
F_vega = 2.56e-6   # lm/m^2
h = np.concatenate((np.linspace(0, 12000, 678), np.full(1200-678,12000)))     # altura del avión, empieza a descender a 30 km de la pista, antes de eso h es cte

# Trayectoria horizontal
x = np.linspace(4000, 50e3, 1200)  # desde 4000m hasta 50 km, un avión empieza su descenso unos 30km antes de la pista, la pista es de hasta 4 km

# Geometría
d = np.sqrt(x**2 + h**2)       # distancia real
cosZ = h / d                  # coseno del ángulo cenital


# Magnitud sin corrección
F = Q / (4 * np.pi * d**2)
m = -2.5 * np.log10(F / F_vega)

# Valores de K
K_values = np.linspace(0.1, 2.5, 10)  # número variable de curvas en función de K

# Gráfica
plt.figure()

for K in K_values:
    M = np.zeros_like(m, dtype=float)

    for i in range(len(cosZ)):
        if cosZ[i] <= 0.22:
            M[i] = m[i] + 4*K         #Corrección para evitar la divergencia de cosZ. Rústico pero ayuda
        else:
            M[i] = m[i] + K / cosZ[i]

    plt.plot(x / 1000, M, label=f"K = {K:.1f}")

# Curva sin corrección
plt.plot(x / 1000, m, 'k--', linewidth=2, label="m (sin corrección)")

# Límite visual
plt.axhline(6, linestyle="--", color="gray", label="Límite ojo humano")
plt.axvline(30, linestyle=":", color="black", label="Inicio del descenso")
plt.axvline(7.8, linestyle=":", color="red", label="Inicio de la corrección por divergencia")

plt.gca().invert_yaxis()
plt.xlabel("Distancia horizontal al final de la pista (km)")
plt.ylabel("Magnitud aparente")
plt.title("Magnitud aparente de un avión en aproximación")
plt.legend()
plt.grid(False)

plt.savefig("Proyectos de Astronomía/Avion/magnitud_avion_aproximacion.png", dpi=300, bbox_inches="tight")

plt.show()
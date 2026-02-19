import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import random

# ==============================
# CONFIGURACIÓN
# ==============================

CARPETA = "Movilidad"

ARCHIVO_VIAJES = os.path.join(CARPETA, "datos_viajes.csv")
ARCHIVO_RUTAS = os.path.join(CARPETA, "rutas_guardadas.csv")

# ==============================
# UTILIDADES
# ==============================

def calcular_minutos(h1, h2):
    fmt = '%H:%M'
    try:
        t1 = datetime.strptime(h1, fmt)
        t2 = datetime.strptime(h2, fmt)
        if t2 < t1:
            t2 += timedelta(days=1)
        return (t2 - t1).total_seconds() / 60
    except:
        return 0

def generar_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

# ==============================
# CARGA Y GUARDADO
# ==============================

def cargar_csv(nombre):
    if os.path.exists(nombre):
        return pd.read_csv(nombre)
    else:
        return pd.DataFrame()

def guardar_csv(df, nombre):
    df.to_csv(nombre, index=False)

# ==============================
# GESTIÓN DE RUTAS
# ==============================

def crear_nueva_ruta(df_rutas):
    nombre = input("Nombre de la nueva ruta: ")
    buses = input("Buses que se toman (separados por coma): ")
    transbordos = int(input("Cantidad de transbordos habituales: "))
    color = generar_color()

    nueva = pd.DataFrame([{
        "Ruta": nombre,
        "Buses": buses,
        "Transbordos": transbordos,
        "Color": color
    }])

    df_rutas = pd.concat([df_rutas, nueva], ignore_index=True)
    guardar_csv(df_rutas, ARCHIVO_RUTAS)

    print(f"Ruta '{nombre}' creada con color {color}")
    return df_rutas


def seleccionar_ruta(df_rutas):
    if df_rutas.empty:
        print("No hay rutas creadas.")
        return None

    print("\nRutas disponibles:")
    for i, r in df_rutas.iterrows():
        print(f"{i} - {r['Ruta']}")

    idx = int(input("Seleccione número de ruta: "))
    return df_rutas.iloc[idx]

# ==============================
# CAPTURA DE VIAJE
# ==============================

def capturar_viaje(ruta_info, df_hist):

    print("\n⚠️ Ingrese TODAS las horas en formato 24 horas (HH:MM).")

    fecha = input("Fecha (DD/MM/AAAA): ")
    fecha_dt = pd.to_datetime(fecha, dayfirst=True)

    dia_semana = input("Día de la semana (Ej: Lunes, Martes, etc.): ")
    tipo = input("Tipo (Ida/Regreso): ")
    h_salida = input("Hora salida (HH:MM, 24h): ")

    categoria = "Habitual"
    if not df_hist.empty:
        misma_ruta = df_hist[df_hist["Ruta"] == ruta_info["Ruta"]]
        if not misma_ruta.empty:
            hora_prom = misma_ruta["Hora_Salida"].mode()
            if not hora_prom.empty:
                diff = abs(calcular_minutos(hora_prom[0], h_salida))
                if diff > 20:
                    categoria = "Horario Especial"

    h_estacion = input("Hora llegada estación (HH:MM, 24h): ")
    t_casa_est = calcular_minutos(h_salida, h_estacion)

    h_bus = input("Hora abordaje primer bus (HH:MM, 24h): ")
    espera = calcular_minutos(h_estacion, h_bus)

    n_trans = int(input("Número de transbordos reales: "))
    for i in range(n_trans):
        h_baj = input(f"Bajada transbordo {i+1} (HH:MM, 24h): ")
        h_sub = input(f"Subida siguiente bus {i+1} (HH:MM, 24h): ")
        espera += calcular_minutos(h_baj, h_sub)

    h_fin = input("Hora bajada último bus (HH:MM, 24h): ")
    h_dest = input("Hora llegada destino (HH:MM, 24h): ")

    t_bus = calcular_minutos(h_bus, h_fin)
    t_final = calcular_minutos(h_fin, h_dest)
    total = t_casa_est + espera + t_bus + t_final

    return {
        "Fecha": fecha_dt,
        "Dia_Semana": dia_semana,
        "Ruta": ruta_info["Ruta"],
        "Tipo": tipo,
        "Hora_Salida": h_salida,
        "Categoria_Horaria": categoria,
        "Casa_Estacion": t_casa_est,
        "Espera": espera,
        "Bus": t_bus,
        "Est_Destino": t_final,
        "Total": total
    }

# ==============================
# ANÁLISIS
# ==============================

def analizar_datos(df, df_rutas):

    if df.empty:
        print("No hay datos.")
        return

    print("\n========= PROMEDIOS Y DESVIACIÓN ESTÁNDAR =========")

    resumen_mean = df.groupby(["Ruta", "Tipo"])[
        ["Casa_Estacion", "Espera", "Bus", "Est_Destino", "Total"]
    ].mean()

    resumen_std = df.groupby(["Ruta", "Tipo"])[
        ["Casa_Estacion", "Espera", "Bus", "Est_Destino", "Total"]
    ].std()

    print("\n--- PROMEDIOS ---")
    print(resumen_mean.round(1))

    print("\n--- DESVIACIÓN ESTÁNDAR ---")
    print(resumen_std.round(1))

    print("\n========= MEJOR RUTA =========")
    mejor = df.groupby("Ruta")["Total"].mean().sort_values()
    print(mejor.round(1))
    print(f"\nRuta más rápida en promedio: {mejor.index[0]}")

    print("\n========= MEJORES HORAS DE SALIDA =========")
    horas_optimas = df.groupby("Hora_Salida")["Total"].mean().sort_values()
    print(horas_optimas.head())

    # Guardar gráficas
    for ruta in df["Ruta"].unique():
        sub = df[df["Ruta"] == ruta]
        color = df_rutas[df_rutas["Ruta"] == ruta]["Color"].values[0]

        plt.figure(figsize=(10, 5))
        plt.title(f"Desglose tiempos - {ruta}")

        plt.bar(sub["Fecha"].astype(str), sub["Casa_Estacion"], color=color, alpha=0.3)
        plt.bar(sub["Fecha"].astype(str), sub["Espera"],
                bottom=sub["Casa_Estacion"], color=color, alpha=0.5)

        bottom2 = sub["Casa_Estacion"] + sub["Espera"]
        plt.bar(sub["Fecha"].astype(str), sub["Bus"],
                bottom=bottom2, color=color, alpha=0.7)

        bottom3 = bottom2 + sub["Bus"]
        plt.bar(sub["Fecha"].astype(str), sub["Est_Destino"],
                bottom=bottom3, color=color)

        plt.xticks(rotation=45)
        plt.ylabel("Minutos")
        plt.tight_layout()

        nombre_archivo = os.path.join(CARPETA, f"grafica_{ruta}.png")
        plt.savefig(nombre_archivo)
        plt.close()

        print(f"Gráfica guardada en: {nombre_archivo}")

# ==============================
# FLUJO PRINCIPAL
# ==============================

def main():

    df_viajes = cargar_csv(ARCHIVO_VIAJES)
    df_rutas = cargar_csv(ARCHIVO_RUTAS)

    while True:

        print("\n==== MENÚ PRINCIPAL ====")
        print("1 - Crear nueva ruta")
        print("2 - Registrar viaje en ruta existente")
        print("3 - Analizar datos")
        print("4 - Salir")

        opcion = input("Seleccione opción: ")

        if opcion == "1":
            df_rutas = crear_nueva_ruta(df_rutas)

        elif opcion == "2":
            ruta_info = seleccionar_ruta(df_rutas)
            if ruta_info is not None:
                nuevo = capturar_viaje(ruta_info, df_viajes)
                nuevo_df = pd.DataFrame([nuevo])

                if df_viajes.empty:
                    df_viajes = nuevo_df
                else:
                    df_viajes = pd.concat([df_viajes, nuevo_df], ignore_index=True)
                guardar_csv(df_viajes, ARCHIVO_VIAJES)

        elif opcion == "3":
            analizar_datos(df_viajes, df_rutas)

        elif opcion == "4":
            break

        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()
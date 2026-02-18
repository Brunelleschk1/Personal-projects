import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def calcular_minutos(h1, h2):
    fmt = '%H:%M'
    try:
        t1 = datetime.strptime(h1, fmt)
        t2 = datetime.strptime(h2, fmt)
        return (t2 - t1).total_seconds() / 60
    except: return 0

def cargar_base_datos(nombre_archivo):
    if os.path.exists(nombre_archivo):
        try:
            df = pd.read_csv(nombre_archivo)
            # Convertimos la columna Fecha asegurando que entienda el formato día/mes
            df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
            return df.dropna(subset=['Fecha']) # Elimina filas con fechas mal formateadas
        except: return pd.DataFrame()
    return pd.DataFrame()

def capturar_tramo(tipo, fecha_str, h_salida_habitual):
    print(f"\n--- Datos de {tipo.upper()} ({fecha_str}) ---")
    h_salida = input(f"Hora salida de {'casa' if tipo == 'Ida' else 'origen'} (HH:MM): ")
    
    # Detección de horario especial (> 20 min de la media)
    es_especial = False
    if h_salida_habitual:
        diff = abs(calcular_minutos(h_salida_habitual, h_salida))
        if diff > 20: es_especial = True

    h_llegada_est = input("Hora llegada estación: ")
    t_casa_est = calcular_minutos(h_salida, h_llegada_est)
    
    h_abordaje = input("Hora abordaje primer transporte: ")
    espera = calcular_minutos(h_llegada_est, h_abordaje)
    
    trans = input("¿Hubo transbordos? (s/n): ").lower() == 's'
    if trans:
        n = int(input("¿Cuántos?: "))
        i = 0
        while i < n:
            h_baj = input(f"Hora bajada transbordo {i+1}: ")
            h_sub = input(f"Hora subida transporte {i+2}: ")
            espera += calcular_minutos(h_baj, h_sub)
            i += 1
            
    h_fin_transporte = input("Hora bajada último transporte: ")
    h_destino = input("Hora llegada destino final: ")
    
    t_bus_neto = calcular_minutos(h_abordaje, h_fin_transporte) - (espera - calcular_minutos(h_llegada_est, h_abordaje))
    t_total = t_casa_est + espera + t_bus_neto + calcular_minutos(h_fin_transporte, h_destino)

    return {
        'Fecha': fecha_str, 'Tipo': tipo, 'Hora_Salida': h_salida,
        'Casa_Estacion': t_casa_est, 'Espera': espera, 'Bus': t_bus_neto,
        'Est_Destino': calcular_minutos(h_fin_transporte, h_destino), 
        'Total': t_total, 'Especial': es_especial
    }

def registrar_jornada(df_hist):
    nuevos = []
    continuar = True
    
    while continuar:
        f_input = input("\nIngrese Fecha (DD/MM/AAAA) o 'salir': ")
        if f_input.lower() == 'salir':
            continuar = False
        else:
            # Obtener horas habituales para comparación
            h_hab_ida = None
            h_hab_reg = None
            if not df_hist.empty:
                h_hab_ida = df_hist[df_hist['Tipo']=='Ida']['Hora_Salida'].mode().max()
                h_hab_reg = df_hist[df_hist['Tipo']=='Regreso']['Hora_Salida'].mode().max()

            # Captura de Ida y Regreso
            nuevos.append(capturar_tramo("Ida", f_input, h_hab_ida))
            nuevos.append(capturar_tramo("Regreso", f_input, h_hab_reg))
            
    return pd.DataFrame(nuevos)

def analizar_y_graficar(df):
    if df.empty:
        print("No hay datos para analizar.")
        return

    # Filtro último mes
    hace_30_dias = datetime.now() - timedelta(days=30)
    df_mes = df[df['Fecha'] >= hace_30_dias].copy()

    for cat in ['Ida', 'Regreso']:
        sub = df_mes[df_mes['Tipo'] == cat]
        if not sub.empty:
            print(f"\nANÁLISIS {cat.upper()} (Últimos 30 días)")
            print(f"Promedio Total: {sub['Total'].mean():.1f} min")
            
            # Gráfico de barras apiladas
            sub.plot(x='Fecha', y=['Casa_Estacion', 'Espera', 'Bus', 'Est_Destino'], 
                     kind='bar', stacked=True, figsize=(10, 4), title=f"Desglose {cat}")
            plt.ylabel("Minutos")
            plt.tight_layout()
            plt.show()

# --- FLUJO PRINCIPAL ---
archivo = 'datos_viajes.csv'
historico = cargar_base_datos(archivo)
nuevas_entradas = registrar_jornada(historico)

if not nuevas_entradas.empty:
    # Convertir fechas de nuevas entradas antes de concatenar
    nuevas_entradas['Fecha'] = pd.to_datetime(nuevas_entradas['Fecha'], dayfirst=True)
    df_final = pd.concat([historico, nuevas_entradas], ignore_index=True)
    df_final.to_csv(archivo, index=False)
else:
    df_final = historico

analizar_y_graficar(df_final)
# probar_modelo.py

import tensorflow as tf
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import MinMaxScaler


# Cargar el modelo entrenado desde el archivo guardado
modelo = tf.keras.models.load_model('financial_model.h5')

# Cargar el scaler guardado
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar los nuevos datos de prueba desde el archivo JSON
with open('clientes_prueba.json', 'r') as f:
    datos_prueba = json.load(f)

# Convertir los datos del JSON en un DataFrame
df_prueba = pd.DataFrame(datos_prueba)

# Seleccionar las columnas que se utilizarán para las predicciones (saldo_actual, ingresos_acumulados, gastos_acumulados)
X_nuevos = df_prueba[['saldo_actual', 'ingresos_acumulados', 'gastos_acumulados']]

# Escalar los nuevos datos con el scaler cargado
X_nuevos_escalado = scaler.transform(X_nuevos)

# Realizar predicciones para cada cliente
predicciones = modelo.predict(X_nuevos_escalado)

# Mostrar las predicciones para cada cliente
for i, prediccion in enumerate(predicciones):
    cliente = df_prueba.loc[i, 'usuario']
    print(f"\nUsuario {cliente} - Quincena {df_prueba.loc[i, 'quincena']} predicción: {prediccion[0]}")
    
    if prediccion > 0.52:
        print(f"Usuario {cliente} - Quincena {df_prueba.loc[i, 'quincena']}: Cumplirá su meta de ahorro.")
    else:
        print(f"Usuario {cliente} - Quincena {df_prueba.loc[i, 'quincena']}: No cumplirá su meta de ahorro.")

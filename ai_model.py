"""Este archivo contiene la lógica para construir 
el modelo de IA y hacer predicciones basadas en 
los datos financieros simulados o reales."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Funcion para cargar y escalar los datos simulados
def cargar_datos():
    datos = pd.read_csv('financial_data.csv')
    X = datos[['balance', 'ingresos', 'gastos']]  # Caracteristicas de entrada
    y = datos['meta_ahorro']  # Meta de ahorro (objetivo)
    
    # Escalar los datos
    scaler = MinMaxScaler()
    X_escalado = scaler.fit_transform(X)
    
    # Dividir en datos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Funcion para construir el modelo de IA
def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(64, input_shape=(3,), activation='relu'))  # Primera capa
    modelo.add(Dense(32, activation='relu'))  # Capa oculta
    modelo.add(Dense(16, activation='relu'))  # Capa oculta
    modelo.add(Dense(1, activation='sigmoid'))  # Capa de salida (prediccion binaria)
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

# Entrenamiento del modelo
def entrenar_modelo():
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos()
    modelo = construir_modelo()
    
    # Entrenar el modelo
    modelo.fit(X_entrenamiento, y_entrenamiento, epochs=50, batch_size=32, validation_data=(X_prueba, y_prueba))
    
    # Evaluar el modelo
    perdida, precision = modelo.evaluate(X_prueba, y_prueba)
    print(f'Precision en los datos de prueba: {precision * 100:.2f}%')
    
    # Guardar el modelo entrenado
    modelo.save('financial_model.h5')
    print("Modelo guardado como financial_model.h5")

if __name__ == "__main__":
    entrenar_modelo()

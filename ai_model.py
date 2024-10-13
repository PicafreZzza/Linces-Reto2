"""""Este archivo contiene la lógica para construir 
el modelo de IA y hacer predicciones basadas en 
los datos financieros simulados o reales."""

#ai_model.py

"""import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import pickle
import json

# Función para cargar y escalar los datos del archivo JSON proporcionado
def cargar_datos():
    with open('datos_financieros_balanceados_500.json', 'r') as f:
        data = json.load(f)
    
    datos = pd.DataFrame(data)
    X = datos[['saldo_actual', 'ingresos_acumulados', 'gastos_acumulados']]
    y = datos['meta_ahorro'].astype(int)
    
    # Escalar los datos
    scaler = MinMaxScaler()
    X_escalado = scaler.fit_transform(X)

    # Guardar el scaler para futuras predicciones
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
    
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Función para construir el modelo de IA con ajustes adicionales
def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(256, input_shape=(3,), kernel_regularizer=l2(0.01)))  # Regularización L2
    modelo.add(LeakyReLU(alpha=0.1))  # Usar LeakyReLU en lugar de ReLU estándar
    modelo.add(Dropout(0.4))  # Reducción del Dropout para prevenir overfitting
    modelo.add(Dense(128, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dense(1, activation='sigmoid'))  # Capa de salida para predicción binaria
    
    # Ajuste del optimizador con un learning rate reducido
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return modelo

# Entrenamiento del modelo con ajuste de hiperparámetros
def entrenar_modelo():
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos()
    modelo = construir_modelo()

    # Implementar Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Añadir una reducción de la tasa de aprendizaje si no mejora la validación
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Entrenar el modelo con más épocas y un tamaño de lote ajustado
    modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, batch_size=32, validation_data=(X_prueba, y_prueba), 
               callbacks=[early_stopping, reduce_lr])
    
    # Evaluar el modelo
    perdida, precision = modelo.evaluate(X_prueba, y_prueba)
    print(f'Precisión en los datos de prueba: {precision * 100:.2f}%')

    # Hacer predicciones con el conjunto de prueba
    y_pred = modelo.predict(X_prueba)
    y_pred_clases = (y_pred > 0.5).astype(int)

    # Matriz de confusión
    print("Matriz de Confusión:")
    print(confusion_matrix(y_prueba, y_pred_clases))

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_prueba, y_pred_clases))

    # Guardar el modelo entrenado
    modelo.save('financial_model_v2.h5')
    print("Modelo guardado como financial_model_v2.h5")

if __name__ == "__main__":
    entrenar_modelo()"""""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import pickle
import mysql.connector

# Configuración de la conexión a MySQL
db_config = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'bank_app'
}

# Función para conectar a la base de datos
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Función para cargar y escalar los datos desde la base de datos
def cargar_datos():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Obtener los datos de los usuarios de la base de datos
    cursor.execute("SELECT saldo_actual, ingresos_acumulados, gastos_acumulados, meta_ahorro FROM users")
    rows = cursor.fetchall()

    # Convertir los datos a un DataFrame de Pandas
    datos = pd.DataFrame(rows)

    # Características y etiquetas
    X = datos[['saldo_actual', 'ingresos_acumulados', 'gastos_acumulados']]
    y = datos['meta_ahorro'].astype(int)

    # Escalar los datos
    scaler = MinMaxScaler()
    X_escalado = scaler.fit_transform(X)

    # Guardar el scaler para futuras predicciones
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Dividir en datos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
    
    cursor.close()
    connection.close()

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Función para construir el modelo de IA
def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(256, input_shape=(3,), kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(128, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return modelo

# Entrenamiento del modelo
def entrenar_modelo():
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos()
    modelo = construir_modelo()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Entrenar el modelo
    modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, batch_size=32, validation_data=(X_prueba, y_prueba), 
               callbacks=[early_stopping, reduce_lr])
    
    # Evaluar el modelo
    perdida, precision = modelo.evaluate(X_prueba, y_prueba)
    print(f'Precisión en los datos de prueba: {precision * 100:.2f}%')

    # Hacer predicciones con el conjunto de prueba
    y_pred = modelo.predict(X_prueba)
    y_pred_clases = (y_pred > 0.5).astype(int)

    # Matriz de confusión
    print("Matriz de Confusión:")
    print(confusion_matrix(y_prueba, y_pred_clases))

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_prueba, y_pred_clases))

    # Guardar el modelo entrenado
    modelo.save('financial_model_v2.h5')
    print("Modelo guardado como financial_model_v2.h5")

if __name__ == "__main__":
    entrenar_modelo()

"""""Este archivo contiene la lógica para construir 
el modelo de IA y hacer predicciones basadas en 
los datos financieros simulados o reales."""

#ai_model.py

"""import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import pickle
import json

# Función para cargar y escalar los datos del archivo JSON proporcionado
def cargar_datos():
    with open('datos_financieros_balanceados_500.json', 'r') as f:
        data = json.load(f)
    
    datos = pd.DataFrame(data)
    X = datos[['saldo_actual', 'ingresos_acumulados', 'gastos_acumulados']]
    y = datos['meta_ahorro'].astype(int)
    
    # Escalar los datos
    scaler = MinMaxScaler()
    X_escalado = scaler.fit_transform(X)

    # Guardar el scaler para futuras predicciones
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
    
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Función para construir el modelo de IA con ajustes adicionales
def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(256, input_shape=(3,), kernel_regularizer=l2(0.01)))  # Regularización L2
    modelo.add(LeakyReLU(alpha=0.1))  # Usar LeakyReLU en lugar de ReLU estándar
    modelo.add(Dropout(0.4))  # Reducción del Dropout para prevenir overfitting
    modelo.add(Dense(128, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dense(1, activation='sigmoid'))  # Capa de salida para predicción binaria
    
    # Ajuste del optimizador con un learning rate reducido
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return modelo

# Entrenamiento del modelo con ajuste de hiperparámetros
def entrenar_modelo():
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos()
    modelo = construir_modelo()

    # Implementar Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Añadir una reducción de la tasa de aprendizaje si no mejora la validación
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Entrenar el modelo con más épocas y un tamaño de lote ajustado
    modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, batch_size=32, validation_data=(X_prueba, y_prueba), 
               callbacks=[early_stopping, reduce_lr])
    
    # Evaluar el modelo
    perdida, precision = modelo.evaluate(X_prueba, y_prueba)
    print(f'Precisión en los datos de prueba: {precision * 100:.2f}%')

    # Hacer predicciones con el conjunto de prueba
    y_pred = modelo.predict(X_prueba)
    y_pred_clases = (y_pred > 0.5).astype(int)

    # Matriz de confusión
    print("Matriz de Confusión:")
    print(confusion_matrix(y_prueba, y_pred_clases))

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_prueba, y_pred_clases))

    # Guardar el modelo entrenado
    modelo.save('financial_model_v2.h5')
    print("Modelo guardado como financial_model_v2.h5")

if __name__ == "__main__":
    entrenar_modelo()"""""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import pickle
import mysql.connector

# Configuración de la conexión a MySQL
db_config = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'bank_app'
}

# Función para conectar a la base de datos
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Función para cargar y escalar los datos desde la base de datos
def cargar_datos():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Obtener los datos de los usuarios de la base de datos
    cursor.execute("SELECT saldo_actual, ingresos_acumulados, gastos_acumulados, meta_ahorro FROM users")
    rows = cursor.fetchall()

    # Convertir los datos a un DataFrame de Pandas
    datos = pd.DataFrame(rows)

    # Características y etiquetas
    X = datos[['saldo_actual', 'ingresos_acumulados', 'gastos_acumulados']]
    y = datos['meta_ahorro'].astype(int)

    # Escalar los datos
    scaler = MinMaxScaler()
    X_escalado = scaler.fit_transform(X)

    # Guardar el scaler para futuras predicciones
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Dividir en datos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
    
    cursor.close()
    connection.close()

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Función para construir el modelo de IA
def construir_modelo():
    modelo = Sequential()
    modelo.add(Dense(256, input_shape=(3,), kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(128, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(64, kernel_regularizer=l2(0.01)))
    modelo.add(LeakyReLU(alpha=0.1))
    modelo.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return modelo

# Entrenamiento del modelo
def entrenar_modelo():
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos()
    modelo = construir_modelo()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Entrenar el modelo
    modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, batch_size=32, validation_data=(X_prueba, y_prueba), 
               callbacks=[early_stopping, reduce_lr])
    
    # Evaluar el modelo
    perdida, precision = modelo.evaluate(X_prueba, y_prueba)
    print(f'Precisión en los datos de prueba: {precision * 100:.2f}%')

    # Hacer predicciones con el conjunto de prueba
    y_pred = modelo.predict(X_prueba)
    y_pred_clases = (y_pred > 0.5).astype(int)

    # Matriz de confusión
    print("Matriz de Confusión:")
    print(confusion_matrix(y_prueba, y_pred_clases))

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_prueba, y_pred_clases))

    # Guardar el modelo entrenado
    modelo.save('financial_model_v2.h5')
    print("Modelo guardado como financial_model_v2.h5")

if __name__ == "__main__":
    entrenar_modelo()

# Proyecto Bancario con Predicción de Meta de Ahorro

## Descripción General

Este proyecto simula una aplicación bancaria que permite a los usuarios ingresar sus datos financieros (ingresos y gastos), ver su saldo actual, historial de transacciones y recibir predicciones sobre si cumplirán o no sus metas de ahorro. También proporciona un plan de acción basado en su situación financiera, la cual se clasifica en cuatro categorías según las predicciones realizadas por un modelo de inteligencia artificial.

## Características

1. **Ingresos y Gastos**: Los usuarios pueden ingresar datos financieros (ingresos y gastos) y calcular el saldo resultante.
2. **Historial de Transacciones**: La aplicación muestra un historial de las transacciones realizadas.
3. **Predicción de Meta de Ahorro**: Utilizando un modelo de inteligencia artificial entrenado con TensorFlow, la aplicación predice si el usuario cumplirá o no su meta de ahorro.
4. **Clasificación Financiera y Plan de Acción**: Según la predicción de la IA, el sistema clasifica la situación financiera del usuario en una de las siguientes categorías:
   - Problemas financieros críticos
   - Situación financiera inestable
   - Salud financiera moderada
   - Situación financiera saludable
5. **Plan de Acción Personalizado**: Dependiendo de la situación financiera del usuario, se le proporciona un plan de acción para mejorar su situación.

## Tecnologías Utilizadas

### Backend
- **Flask**: Framework ligero de Python utilizado para manejar las rutas y la lógica de la aplicación.
- **MySQL**: Base de datos utilizada para almacenar las transacciones, datos de usuarios y sus movimientos financieros.
- **MySQL Connector (mysql.connector)**: Utilizado para conectar Flask con la base de datos MySQL.
- **TensorFlow**: Biblioteca utilizada para construir y entrenar el modelo de inteligencia artificial encargado de realizar las predicciones.
- **Scikit-learn**: Utilizada para escalar los datos financieros antes de ser procesados por el modelo de IA.

### Frontend
- **HTML**: Para la estructura de la interfaz de usuario.
- **CSS/Bootstrap**: Utilizado para el diseño responsivo de la interfaz de usuario.
- **JavaScript**: Para manejar interacciones dinámicas en el frontend.
- **Font Awesome**: Para incorporar iconos en la interfaz.

### Bibliotecas de Python
- **pymysql**: Para gestionar la conexión con la base de datos MySQL.
- **Numpy**: Para operaciones numéricas y manipulación de datos.
- **Pickle**: Para cargar y guardar el scaler y el modelo de IA entrenado.
- **Flask Flash**: Para mostrar mensajes al usuario, como errores y confirmaciones de acciones.

## Estructura del Proyecto

```bash
project-root/
│
├── static/
│   └── img/  # Aquí se almacenan imágenes, como el logo de Banorte
│
├── templates/
│   └── index.html  # La interfaz principal de la aplicación
│
├── app.py  # El archivo principal del backend de la aplicación
├── ai_model.py  # Archivo con la lógica para entrenar y hacer predicciones con el modelo de IA
├── scaler.pkl  # Scaler entrenado para escalar los datos de los usuarios
├── financial_model_v2.h5  # Modelo entrenado de IA
├── README.md  # Este archivo
└── requirements.txt  # Las dependencias necesarias para ejecutar el proyecto

from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
import mysql.connector
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_flash'

# Configuración de la conexión a MySQL
db_config = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'bank_app'
}

# Obtener conexión a la base de datos
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('financial_model_v2.h5')

# Cargar el scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    connection = get_db_connection()
    cursor = connection.cursor()

    # Obtenemos el saldo actual sumando todas las transacciones
    cursor.execute("SELECT SUM(amount) FROM transactions")
    balance = cursor.fetchone()[0] or 0

    # Obtenemos el historial de transacciones
    cursor.execute("SELECT amount, date FROM transactions ORDER BY date DESC")
    transactions = cursor.fetchall()

    # Obtener los últimos ingresos y gastos para hacer la predicción
    cursor.execute("SELECT ingresos, gastos FROM users ORDER BY id DESC LIMIT 1")
    user_data = cursor.fetchone()

    if user_data:
        ingresos = user_data[0]
        gastos = user_data[1]
        saldo_actual = balance
        ingresos_acumulados = ingresos
        gastos_acumulados = gastos

        # Escalar los datos
        X_nuevos = scaler.transform([[saldo_actual, ingresos_acumulados, gastos_acumulados]])

        # Realizar la predicción
        prediccion = modelo.predict(X_nuevos)[0][0]
        prediccion_meta_ahorro = "Sí" if prediccion > 0.5 else "No"

        # Clasificar la situación financiera con varios planes de acción
        plan_accion = []
        if prediccion <= 0.25:
            situacion = "Problemas financieros críticos"
            plan_accion = [
                "Reducción drástica de gastos: eliminar gastos innecesarios.",
                "Generación de ingresos adicionales: buscar más fuentes de ingreso.",
                "Plan de emergencia: crear un fondo de emergencia.",
                "Asesoría financiera: buscar ayuda para reestructurar deudas."
            ]
        elif 0.26 <= prediccion <= 0.50:
            situacion = "Situación financiera inestable"
            plan_accion = [
                "Crear un fondo de emergencia de 3 a 6 meses de gastos esenciales.",
                "Mejora del manejo de deudas.",
                "Ajuste de gastos y planificación a corto plazo."
            ]
        elif 0.51 <= prediccion <= 0.75:
            situacion = "Salud financiera moderada"
            plan_accion = [
                "Optimizar el fondo de emergencia para cubrir de 6 a 12 meses.",
                "Diversificación de inversiones.",
                "Planificación a largo plazo."
            ]
        else:
            situacion = "Situación financiera saludable"
            plan_accion = [
                "Maximizar las inversiones.",
                "Planificación de la jubilación.",
                "Protección del patrimonio y filantropía."
            ]
    else:
        prediccion_meta_ahorro = "No hay suficientes datos"
        situacion = "No hay suficientes datos"
        plan_accion = ["No se puede generar un plan de acción."]

    cursor.close()
    connection.close()

    return render_template('index.html', balance=balance, transactions=transactions, prediccion_meta_ahorro=prediccion_meta_ahorro, situacion=situacion, plan_accion=plan_accion)

@app.route('/insert_data', methods=['POST'])
def insert_data():
    # Recibe los datos enviados desde el formulario
    ingresos = float(request.form['ingresos'])
    gastos = float(request.form['gastos'])

    # Obtener el saldo actual desde la base de datos
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT SUM(amount) FROM transactions")
    saldo_actual = cursor.fetchone()[0] or 0
    
    # Convertir saldo_actual a float si es Decimal
    saldo_actual = float(saldo_actual)

    # Calcular el nuevo balance (saldo actual + ingresos - gastos)
    balance_nuevo = saldo_actual + (ingresos - gastos)

    # Verificar si el saldo es negativo
    if balance_nuevo < 0:
        flash("El saldo no puede ser negativo. Transacción no permitida.", "error")
        return redirect(url_for('index'))

    # Insertar la nueva transacción en la base de datos
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO transactions (amount, date) VALUES (%s, %s)", ((ingresos - gastos), date))

    # Actualizar los ingresos y gastos acumulados en la tabla 'users'
    cursor.execute("INSERT INTO users (ingresos, gastos, balance) VALUES (%s, %s, %s)", 
                   (ingresos, gastos, balance_nuevo))

    connection.commit()
    cursor.close()
    connection.close()

    flash("Transacción agregada con éxito.", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

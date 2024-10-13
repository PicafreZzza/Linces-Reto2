CREATE DATABASE IF NOT EXISTS bank_app;

USE bank_app;

-- Crear tabla para las transacciones
CREATE TABLE IF NOT EXISTS transactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    amount DECIMAL(10, 2) NOT NULL,
    date DATETIME NOT NULL
);

-- Crear tabla para almacenar los datos de los usuarios
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    quincena INT NOT NULL,
    ingresos DECIMAL(10, 2) NOT NULL,
    gastos DECIMAL(10, 2) NOT NULL,
    balance DECIMAL(10, 2) NOT NULL,
    saldo_actual DECIMAL(10, 2) NOT NULL,
    ingresos_acumulados DECIMAL(10, 2) NOT NULL,
    gastos_acumulados DECIMAL(10, 2) NOT NULL,
    meta_ahorro TINYINT(1) NOT NULL  -- 1 para 'cumplirá' y 0 para 'no cumplirá'
);

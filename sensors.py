
"""Leitura de sensores: bateria, temperatura, IMU.

Ajuste conforme seu hardware:
- Bateria: ADC (ex.: MCP3008) ou medidor externo via I2C/UART.
- Temperatura: termistor, DS18B20, ou CPU como proxy.
- IMU: MPU6050/BNO055 via I2C.

Se não houver sensores, retornos simulados para não quebrar a missão.
"""
import os
import time
import random

# Exemplos de integrações comentadas:
# import smbus  # I2C

def read_battery_level():
    """Retorna nível de bateria (%). Substitua por leitura real."""
    try:
        # TODO: ler ADC / sensor real
        return 85.0
    except Exception:
        return 85.0

def read_temperature():
    """Retorna temperatura (°C). Substitua por sensor real ou CPU temp."""
    try:
        # Exemplo: CPU temp em Linux: /sys/class/thermal/thermal_zone0/temp (millicelsius)
        path = '/sys/class/thermal/thermal_zone0/temp'
        if os.path.exists(path):
            with open(path,'r') as f:
                mc = int(f.read().strip())
            return round(mc/1000.0, 2)
        # Sem sensor, retorna estimativa
        return -10.0
    except Exception:
        return -10.0

def read_imu():
    """Retorna (gyro, accel) cada um com 3 eixos. Substitua por I2C real."""
    try:
        # TODO: ler IMU real (MPU6050/BNO055). Aqui: simulado.
        gyro = [0.01, -0.02, 0.00]
        accel = [0.10, 0.00, 9.80]
        return gyro, accel
    except Exception:
        return [0,0,0], [0,0,0]

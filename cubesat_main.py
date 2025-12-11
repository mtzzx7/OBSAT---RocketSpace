
"""
CubeSat Payload + Telemetria — Operação a 30 km (balão estratosférico / OBSAT)

Funcionalidades:
- Captura de imagem (OpenCV) e armazenamento em memória (SD/flash).
- Processamento de branqueamento com bleach_detector (mono ou bi-temporal).
- Telemetria via HTTP (Wi-Fi): bateria, temperatura, IMU (gyro/accel), payload.
- Envio por 2 horas, em intervalos de 4 minutos, com payload ≤ 90 bytes.
"""
import os
import time
import json
from datetime import datetime
import traceback

import cv2
import numpy as np

# Módulo de detecção (arquivos entregues previamente)
from bleach_detector import percent_bleached_single, percent_bleached_change

from telemetry import send_json, build_packet
from sensors import read_battery_level, read_temperature, read_imu
from camera_capture import capture_image

# ---------- Utilitários ----------
TS_FMT = "%Y-%m-%dT%H:%M:%S"

def now_ts_str():
    return datetime.utcnow().strftime(TS_FMT)

# ---------- Loop principal ----------

def run_mission():
    # Carregar configuração
    with open('config.json','r',encoding='utf-8') as f:
        cfg = json.load(f)
    sat_id = cfg.get('satellite_id','SAT-001')
    total_minutes = int(cfg.get('duration_minutes', 120))
    interval_minutes = int(cfg.get('interval_minutes', 4))
    out_dir = cfg.get('output_dir','data')
    os.makedirs(out_dir, exist_ok=True)

    # Captura inicial (ANTES) — opcional para bi-temporal
    print('[INFO] Iniciando missão...')
    img_before_path = os.path.join(out_dir, f"before_{int(time.time())}.jpg")
    img_before = capture_image(img_before_path, device_index=cfg.get('camera_index',0),
                               width=cfg.get('image_width',1280), height=cfg.get('image_height',720))
    has_before = img_before is not None

    start = time.time()
    cycles = int(total_minutes / interval_minutes)
    for i in range(cycles):
        print(f"\n[INFO] Ciclo {i+1}/{cycles} — {now_ts_str()} ")
        # Captura imagem corrente
        img_path = os.path.join(out_dir, f"img_{i+1}_{int(time.time())}.jpg")
        img_bgr = capture_image(img_path, device_index=cfg.get('camera_index',0),
                                width=cfg.get('image_width',1280), height=cfg.get('image_height',720))
        payload_info = {}
        try:
            if img_bgr is not None:
                # Processamento mono-temporal por padrão
                res = percent_bleached_single(img_bgr)
                percent = res['percent_bleached']
                mask_path = os.path.join(out_dir, os.path.basename(img_path).replace('.jpg','_mask.png'))
                cv2.imwrite(mask_path, res['mask'])
                payload_info = {
                    'bp': round(percent,2),
                    'id': os.path.basename(img_path)
                }
                print(f"[INFO] Branqueamento estimado: {percent:.2f}% | {os.path.basename(img_path)}")

                # Se houver imagem 'antes', calcula mudança
                if has_before and cfg.get('use_change_mode', True):
                    res_ch = percent_bleached_change(img_before, img_bgr)
                    percent_ch = res_ch['percent_bleached_change']
                    mask_ch_path = os.path.join(out_dir, os.path.basename(img_path).replace('.jpg','_mask_change.png'))
                    cv2.imwrite(mask_ch_path, res_ch['mask_change'])
                    payload_info['bc'] = round(percent_ch,2)  # bleaching change
                    print(f"[INFO] Mudança para branqueamento: {percent_ch:.2f}%")
                    # Atualiza o 'antes' para próximo ciclo
                    img_before = img_bgr.copy()
            else:
                print('[WARN] Falha ao capturar imagem.')
                payload_info = {'bp': -1.0, 'id': 'NOIMG'}
        except Exception as e:
            print('[ERROR] Falha no processamento:', e)
            traceback.print_exc()
            payload_info = {'bp': -1.0, 'id': 'ERR'}

        # Ler sensores
        bat = read_battery_level()
        tmp = read_temperature()
        gyro, accel = read_imu()

        # Montar pacote JSON (campos mínimos, payload ≤ 90 bytes)
        pkt = build_packet(
            satellite_id=sat_id,
            battery=bat,
            temperature=tmp,
            gyro=gyro,
            accel=accel,
            payload_info=payload_info,
        )
        print('[INFO] JSON (compacto):', pkt)

        # Enviar
        try:
            ok, status_code = send_json(pkt)
            print(f"[INFO] Telemetria enviada? {ok} (HTTP {status_code})")
        except Exception as e:
            print('[WARN] Falha no envio HTTP:', e)

        # Esperar até o próximo ciclo
        if i < cycles - 1:
            sleep_s = interval_minutes * 60
            print(f"[INFO] Aguardando {interval_minutes} min...")
            time.sleep(sleep_s)

    print('[INFO] Missão concluída.')

if __name__ == '__main__':
    run_mission()


"""Telemetria HTTP (Wi-Fi) com JSON compacto e payload ≤ 90 bytes."""
import json
import time
import os

# Tenta usar 'requests'; se não existir, usa urllib
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    import urllib.request
    HAVE_REQUESTS = False

# Carrega config
with open('config.json','r',encoding='utf-8') as f:
    CFG = json.load(f)

ENDPOINT_URL = CFG.get('endpoint_url', 'http://127.0.0.1:8000/telemetry')
TEAM_ID = CFG.get('team_id','TEAM-XX')
MISSION_ID = CFG.get('mission_id','MISSION-YY')
TIMEOUT = int(CFG.get('http_timeout', 10))

def _compact_float(x, decimals=2):
    try:
        return round(float(x), decimals)
    except Exception:
        return x

def build_packet(satellite_id, battery, temperature, gyro, accel, payload_info):
    """
    Constrói JSON **compacto**, com 'pl' (payload) ≤ 90 bytes.
    - Arrays (gyro/accel) viram strings com 2 casas decimais: "gx,gy,gz".
    - 'pl' é string compacta: "bp=12.5;id=IMG_001;bc=10.0" (se existir bc).
    """
    gx = ','.join([f"{_compact_float(v)}" for v in (gyro or [0,0,0])])
    ax = ','.join([f"{_compact_float(v)}" for v in (accel or [0,0,0])])

    # Monta payload compactado
    bp = payload_info.get('bp')
    img_id = payload_info.get('id','')
    bc = payload_info.get('bc', None)
    parts = [f"bp={_compact_float(bp)}", f"id={img_id}"]
    if bc is not None:
        parts.append(f"bc={_compact_float(bc)}")
    pl = ';'.join(parts)

    # Garante limite de 90 bytes
    pl_bytes = pl.encode('utf-8')
    if len(pl_bytes) > 90:
        # corta id
        max_id_len = max(0, 90 - len(f"bp={_compact_float(bp)};bc={_compact_float(bc)};id=") )
        img_id_short = img_id[:max_id_len]
        pl = f"bp={_compact_float(bp)};id={img_id_short}"
        if bc is not None:
            pl = f"bp={_compact_float(bp)};bc={_compact_float(bc)};id={img_id_short}"

    pkt = {
        'team': TEAM_ID,
        'mission': MISSION_ID,
        'sat': satellite_id,
        'ts': int(time.time()),
        'bat': _compact_float(battery),
        'tmp': _compact_float(temperature),
        'g': gx,
        'a': ax,
        'pl': pl
    }
    return json.dumps(pkt, separators=(',',':'))  # sem espaços

def send_json(packet_str):
    """Envia JSON via HTTP POST. Retorna (ok, status_code)."""
    headers = {'Content-Type': 'application/json'}
    if HAVE_REQUESTS:
        r = requests.post(ENDPOINT_URL, data=packet_str, headers=headers, timeout=TIMEOUT)
        return (200 <= r.status_code < 300), r.status_code
    else:
        req = urllib.request.Request(ENDPOINT_URL, data=packet_str.encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return (200 <= resp.status < 300), resp.status

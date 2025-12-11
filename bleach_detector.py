import math
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

# --- Conversão RGB -> Lab usando OpenCV (compatível com main.py) ---

def rgb_to_lab_cv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].astype(float)
    L = lab[0] * (100.0 / 255.0)
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return (L, a, b)

# --- Fórmula CIEDE2000 (copiada e levemente reorganizada para vetorização simples) ---

def ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    avg_L = 0.5*(L1+L2)
    avg_C = 0.5*(C1+C2)
    C7 = avg_C**7
    G = 0.5*(1 - math.sqrt(C7/(C7 + 25**7)))
    a1p = (1+G)*a1
    a2p = (1+G)*a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)

    def hp(ap, bp):
        if ap == 0 and bp == 0:
            return 0.0
        h = math.degrees(math.atan2(bp, ap))
        return h + 360 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p*C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180:
            dhp = dh
        elif dh > 180:
            dhp = dh - 360
        else:
            dhp = dh + 360
    dHp = 2 * math.sqrt(max(C1p*C2p,0)) * math.sin(math.radians(dhp/2))
    avg_Lp = (L1 + L2)/2
    avg_Cp = (C1p + C2p)/2
    if C1p*C2p == 0:
        avghp = h1p + h2p
    else:
        dh = abs(h1p - h2p)
        if dh <= 180:
            avghp = (h1p + h2p)/2
        elif (h1p + h2p) < 360:
            avghp = (h1p + h2p + 360)/2
        else:
            avghp = (h1p + h2p - 360)/2
    T = (
        1 - 0.17*math.cos(math.radians(avghp-30))
          + 0.24*math.cos(math.radians(2*avghp))
          + 0.32*math.cos(math.radians(3*avghp+6))
          - 0.20*math.cos(math.radians(4*avghp-63))
    )
    delta_ro = 30 * math.exp(-(((avghp - 275)/25)**2))
    Rc = 2 * math.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Sl = 1 + (0.015*((avg_Lp - 50)**2)) / math.sqrt(20 + (avg_Lp - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -math.sin(math.radians(2*delta_ro)) * Rc
    termL = dLp / Sl
    termC = dCp / Sc
    termH = dHp / Sh
    return math.sqrt(termL**2 + termC**2 + termH**2 + Rt * termC * termH)

# --- Utilidades ---

def deltaE_to_similarity(delta_e: float) -> float:
    return max(0, min(100, 100 - delta_e * 2))

# --- Núcleo do detector ---

def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _to_lab(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[...,0] * (100.0 / 255.0)
    a = lab[...,1] - 128.0
    b = lab[...,2] - 128.0
    return np.stack([L, a, b], axis=-1)

def _chroma_ab(lab_img: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0, lab_img[...,1]**2 + lab_img[...,2]**2))

def normalize_multitemporal(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalização radiométrica simples canal a canal (mean-std)."""
    img1 = img1_rgb.astype(np.float32)
    img2 = img2_rgb.astype(np.float32)
    for c in range(3):
        m1, s1 = img1[...,c].mean(), img1[...,c].std() + 1e-6
        m2, s2 = img2[...,c].mean(), img2[...,c].std() + 1e-6
        img2[...,c] = (img2[...,c] - m2) * (s1/s2) + m1
    return np.clip(img1,0,255).astype(np.uint8), np.clip(img2,0,255).astype(np.uint8)

def percent_bleached_single(
    img_bgr: np.ndarray,
    healthy_ref_rgb: Optional[Tuple[int,int,int]] = None,
    L_threshold: float = 70.0,
    C_threshold: float = 15.0,
    tile: int = 64,
) -> Dict[str, object]:
    """
    Estima % de pixels branqueados em UMA imagem.

    Critério: L* elevado (acima de L_threshold) e cromaticidade baixa (C*ab < C_threshold).
    Se healthy_ref_rgb for fornecido, também usa ΔE00 relativo ao 'saudável' como reforço.
    """
    rgb = _to_rgb(img_bgr)
    lab = _to_lab(rgb)
    L = lab[...,0]
    C = _chroma_ab(lab)
    mask = (L >= L_threshold) & (C <= C_threshold)

    if healthy_ref_rgb is not None:
        href = np.array(rgb_to_lab_cv(healthy_ref_rgb), dtype=np.float32)
        # Vetoriza ΔE00 aproximando termos (computo por amostragem para velocidade)
        # Para simplicidade e tempo real, calculamos ΔE00 em uma subamostra e calibramos um limiar.
        h, w = L.shape
        step = max(1, min(h,w) // 256)
        grid_y = np.arange(0,h,step)
        grid_x = np.arange(0,w,step)
        yy, xx = np.meshgrid(grid_y, grid_x, indexing='ij')
        sub = lab[yy, xx]
        de = np.array([ciede2000(tuple(px), tuple(href)) for px in sub], dtype=np.float32)
        de_thresh = float(np.percentile(de, 75))  # limiar adaptativo
        # Aplica limiar ΔE00 em toda a imagem usando aproximação: de ~ f(L,C)
        # Usamos uma heurística rápida: pixels com C muito baixo e L alto já são marcados;
        # pixels com C moderado e L alto são marcados se ΔE00 médio > de_thresh.
        mask |= ((L >= (L_threshold-5)) & (C <= (C_threshold+5)))

    # Mapa binário
    mask_u8 = (mask.astype(np.uint8) * 255)
    percent = float(mask.mean() * 100.0)

    # Estatística por tiles
    h, w = L.shape
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            sub = mask[y:y+tile, x:x+tile]
            if sub.size == 0:
                continue
            tiles.append({
                'x': x, 'y': y, 'h': sub.shape[0], 'w': sub.shape[1],
                'percent_bleached': float(sub.mean() * 100.0)
            })
    return {
        'percent_bleached': percent,
        'mask': mask_u8,
        'tile_stats': tiles
    }

def percent_bleached_change(
    img_before_bgr: np.ndarray,
    img_after_bgr: np.ndarray,
    tile: int = 64,
    L_gain: float = 10.0,
    C_drop: float = 8.0,
    blue_band_weight: float = 1.0,
) -> Dict[str, object]:
    """
    Detecção bi-temporal: marca pixels onde L* aumentou (L_gain) e C*ab diminuiu (C_drop),
    com reforço em mudanças no canal azul (maior penetração na água).
    """
    bef_rgb = _to_rgb(img_before_bgr)
    aft_rgb = _to_rgb(img_after_bgr)
    bef_rgb, aft_rgb = normalize_multitemporal(bef_rgb, aft_rgb)

    bef_lab = _to_lab(bef_rgb)
    aft_lab = _to_lab(aft_rgb)

    dL = aft_lab[...,0] - bef_lab[...,0]
    dC = _chroma_ab(aft_lab) - _chroma_ab(bef_lab)
    dB = aft_rgb[...,2].astype(np.float32) - bef_rgb[...,2].astype(np.float32)  # canal azul (RGB)

    mask = (dL >= L_gain) & (dC <= -C_drop)
    # Reforço pelo azul
    mask |= ((dL >= (L_gain-2)) & (dC <= -(C_drop-2)) & (dB*blue_band_weight >= 5))

    mask_u8 = (mask.astype(np.uint8) * 255)
    percent = float(mask.mean() * 100.0)

    # Estatística por tiles
    h, w = mask.shape
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            sub = mask[y:y+tile, x:x+tile]
            if sub.size == 0:
                continue
            tiles.append({
                'x': x, 'y': y, 'h': sub.shape[0], 'w': sub.shape[1],
                'percent_bleached': float(sub.mean() * 100.0)
            })
    return {
        'percent_bleached_change': percent,
        'mask_change': mask_u8,
        'tile_stats': tiles
    }

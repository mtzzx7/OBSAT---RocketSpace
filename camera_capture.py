
"""Captura de imagem com OpenCV. Ajuste device_index, resolução e foco.
Use CSI (PiCam) ou USB webcam. Em SBC sem câmera, retorna None.
"""
import cv2

def capture_image(out_path, device_index=0, width=1280, height=720):
    try:
        cap = cv2.VideoCapture(device_index)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        cv2.imwrite(out_path, frame)
        return frame
    except Exception:
        return None


# app_bleach_tk.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ---------- Utilidades de cor (Lab) ----------
def rgb_to_lab_img(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[...,0] * (100.0 / 255.0)
    a = lab[...,1] - 128.0
    b = lab[...,2] - 128.0
    return np.stack([L, a, b], axis=-1)

def chroma_ab(lab_img):
    return np.sqrt(np.maximum(0, lab_img[...,1]**2 + lab_img[...,2]**2))

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def normalize_pair(img1_rgb, img2_rgb):
    i1 = img1_rgb.astype(np.float32)
    i2 = img2_rgb.astype(np.float32)
    for c in range(3):
        m1, s1 = i1[...,c].mean(), i1[...,c].std() + 1e-6
        m2, s2 = i2[...,c].mean(), i2[...,c].std() + 1e-6
        i2[...,c] = (i2[...,c] - m2) * (s1/s2) + m1
    return np.clip(i1,0,255).astype(np.uint8), np.clip(i2,0,255).astype(np.uint8)

# ---------- Núcleo de detecção ----------
def detect_single(img_bgr, L_threshold=70.0, C_threshold=15.0):
    """Retorna mask (uint8) e percent (%) de provável branqueamento em UMA imagem."""
    rgb = to_rgb(img_bgr)
    lab = rgb_to_lab_img(rgb)
    L = lab[...,0]
    C = chroma_ab(lab)
    mask = (L >= L_threshold) & (C <= C_threshold)
    return (mask.astype(np.uint8)*255), float(mask.mean()*100.0)

def detect_change(img_before_bgr, img_after_bgr, L_gain=10.0, C_drop=8.0, blue_weight=1.0):
    """Retorna mask_change (uint8) e percent (%) de mudança para branqueamento (bi-temporal)."""
    b_rgb = to_rgb(img_before_bgr)
    a_rgb = to_rgb(img_after_bgr)
    b_rgb, a_rgb = normalize_pair(b_rgb, a_rgb)
    b_lab = rgb_to_lab_img(b_rgb)
    a_lab = rgb_to_lab_img(a_rgb)
    dL = a_lab[...,0] - b_lab[...,0]
    dC = chroma_ab(a_lab) - chroma_ab(b_lab)
    dB = a_rgb[...,2].astype(np.float32) - b_rgb[...,2].astype(np.float32)  # azul (RGB)
    mask = (dL >= L_gain) & (dC <= -C_drop)
    mask |= ((dL >= (L_gain-2)) & (dC <= -(C_drop-2)) & (dB*blue_weight >= 5))
    return (mask.astype(np.uint8)*255), float(mask.mean()*100.0)

# ---------- GUI ----------
class BleachApp:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("Protótipo — Detecção de Branqueamento de Corais (Tkinter)")
        self.app.geometry("1080x720")
        self.app.configure(bg="#2b2b2b")
        self.app.resizable(False, False)

        self.left = tk.Frame(self.app, bg="#2b2b2b")
        self.left.pack(side="left", padx=12, pady=12)
        self.right = tk.Frame(self.app, bg="#2b2b2b")
        self.right.pack(side="left", padx=12, pady=12)

        # Canvas principal (duas imagens lado a lado)
        self.canvas = tk.Canvas(self.left, width=900, height=600, bg="#1e1e1e")
        self.canvas.pack()
        # Botões
        btns = tk.Frame(self.left, bg="#2b2b2b")
        btns.pack(pady=8)
        tk.Button(btns, text="📂 Carregar ANTES", command=self.load_before, bg="#4a90e2", fg="white").grid(row=0, column=0, padx=6)
        tk.Button(btns, text="📂 Carregar DEPOIS", command=self.load_after,  bg="#4a90e2", fg="white").grid(row=0, column=1, padx=6)
        tk.Button(btns, text="🧪 Detectar (Uma Imagem)", command=self.run_single, bg="#00bfa5", fg="white").grid(row=0, column=2, padx=6)
        tk.Button(btns, text="🔄 Detectar (Antes→Depois)", command=self.run_change, bg="#00bfa5", fg="white").grid(row=0, column=3, padx=6)
        tk.Button(btns, text="💾 Salvar Máscara", command=self.save_mask, bg="#8e24aa", fg="white").grid(row=0, column=4, padx=6)

        self.status = tk.Label(self.left, text="Aguardando imagens...", bg="#2b2b2b", fg="white")
        self.status.pack(pady=6)

        # Painel lateral
        tk.Label(self.right, text="Resultados", bg="#2b2b2b", fg="white", font=("Arial", 14, "bold")).pack()
        self.sim_canvas = tk.Canvas(self.right, width=220, height=60, bg="#111")
        self.sim_canvas.pack(pady=8)
        self.info = tk.Text(self.right, width=40, height=20, bg="#111", fg="white")
        self.info.pack()

        # Estado
        self.img_b = None  # BGR "antes" ou única
        self.img_a = None  # BGR "depois"
        self.photo_before = None
        self.photo_after = None
        self.overlay_mask = None  # última máscara
        self.last_percent = None

    def load_before(self):
        fp = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not fp: return
        self.img_b = cv2.imread(fp)
        self.draw_images()
        self.status.config(text="Imagem ANTES carregada.")

    def load_after(self):
        fp = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not fp: return
        self.img_a = cv2.imread(fp)
        self.draw_images()
        self.status.config(text="Imagem DEPOIS carregada.")

    def draw_images(self):
        self.canvas.delete("all")
        # desenha até duas imagens, dimensionando para caber 900x600
        def prep(img_bgr, max_w, max_h):
            rgb = to_rgb(img_bgr)
            h, w = rgb.shape[:2]
            scale = min(max_w / w, max_h / h)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(rgb, (nw, nh))
            return ImageTk.PhotoImage(Image.fromarray(resized)), nw, nh
        xoff = 10
        yoff = 10
        if self.img_b is not None:
            self.photo_before, bw, bh = prep(self.img_b, 440, 580)
            self.canvas.create_image(xoff, yoff, anchor="nw", image=self.photo_before)
            self.canvas.create_text(xoff + bw//2, yoff + bh + 12, text="ANTES/ÚNICA", fill="white")
            xoff += bw + 20
        if self.img_a is not None:
            self.photo_after, aw, ah = prep(self.img_a, 440, 580)
            self.canvas.create_image(xoff, yoff, anchor="nw", image=self.photo_after)
            self.canvas.create_text(xoff + aw//2, yoff + ah + 12, text="DEPOIS", fill="white")

        # Se houver overlay, redesenha por cima do primeiro painel
        if self.overlay_mask is not None and self.img_b is not None:
            # cria overlay do tamanho exibido
            rgb = to_rgb(self.img_b)
            h, w = rgb.shape[:2]
            scale = min(440 / w, 580 / h)
            nw, nh = int(w*scale), int(h*scale)
            mask_small = cv2.resize(self.overlay_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            # cor vermelha semitransparente
            overlay = np.zeros((nh, nw, 4), dtype=np.uint8)
            overlay[...,0:3] = (255, 0, 0)    # vermelho
            overlay[...,3] = (mask_small > 0).astype(np.uint8) * 120  # alpha
            pil_over = Image.fromarray(overlay, mode="RGBA")
            self.canvas.create_image(10, 10, anchor="nw", image=ImageTk.PhotoImage(pil_over))

    def run_single(self):
        if self.img_b is None:
            messagebox.showinfo("Info", "Carregue ao menos uma imagem (ANTES/ÚNICA).")
            return
        mask, percent = detect_single(self.img_b)
        self.overlay_mask = mask
        self.last_percent = percent
        self.update_info(percent, mode="single")
        self.draw_images()

    def run_change(self):
        if self.img_b is None or self.img_a is None:
            messagebox.showinfo("Info", "Carregue as duas imagens (ANTES e DEPOIS).")
            return
        mask, percent = detect_change(self.img_b, self.img_a)
        self.overlay_mask = mask
        self.last_percent = percent
        self.update_info(percent, mode="change")
        self.draw_images()

    def update_info(self, percent, mode="single"):
        self.info.config(state="normal")
        self.info.delete("1.0", tk.END)
        title = "% Branqueado" if mode == "single" else "% Mudança para Branqueamento"
        self.info.insert(tk.END, f"{title}: {percent:.2f}%\n\n")
        # barra
        self.sim_canvas.delete("all")
        bar_w = int((percent/100.0)*220)
        color = "#00C853" if percent>80 else "#FFB300" if percent>50 else "#D50000"
        self.sim_canvas.create_rectangle(0, 0, bar_w, 20, fill=color, outline="")
        self.sim_canvas.create_text(110, 35, text=f"{percent:.2f}%", fill="white")
        self.info.config(state="disabled")
        self.status.config(text=f"{title}: {percent:.2f}%")

    def save_mask(self):
        if self.overlay_mask is None:
            messagebox.showinfo("Info", "Nenhuma máscara gerada ainda.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not fp: return
        cv2.imwrite(fp, self.overlay_mask)
        messagebox.showinfo("Salvo", f"Máscara salva em:\n{fp}")

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    BleachApp().run()

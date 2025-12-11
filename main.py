import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import math

# -------------------- UTILIDADES DE COR --------------------
def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)

def rgb_tuple(cl):
    return (int(cl[0]), int(cl[1]), int(cl[2]))

# --- Conversão RGB -> Lab usando OpenCV ---
def rgb_to_lab_cv(rgb):
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].astype(float)
    L = lab[0] * (100.0 / 255.0)
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return (L, a, b)

# -------------------- Fórmula CIEDE2000 --------------------
def ciede2000(lab1, lab2):
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
        1 -
        0.17*math.cos(math.radians(avghp-30)) +
        0.24*math.cos(math.radians(2*avghp)) +
        0.32*math.cos(math.radians(3*avghp+6)) -
        0.20*math.cos(math.radians(4*avghp-63))
    )

    delta_ro = 30 * math.exp(-((avghp - 275)/25)**2)
    Rc = 2 * math.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))

    Sl = 1 + (0.015*((avg_Lp - 50)**2)) / math.sqrt(20 + (avg_Lp - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -math.sin(math.radians(2*delta_ro)) * Rc

    termL = dLp / Sl
    termC = dCp / Sc
    termH = dHp / Sh

    return math.sqrt(termL**2 + termC**2 + termH**2 + Rt * termC * termH)

def deltaE_to_similarity(delta_e):
    return max(0,min(100,100 - delta_e * 2))

# -------------------- Lista de cores nomeadas --------------------
NAMED_COLORS = {
    "red": (255,0,0), "darkred": (139,0,0), "tomato": (255,99,71),
    "yellow": (255,255,0), "gold": (255,215,0), "lightyellow": (255,255,224),
    "orange": (255,165,0), "pink": (255,192,203), "deeppink": (255,20,147),
    "magenta": (255,0,255), "purple": (128,0,128), "blue": (0,0,255),
    "skyblue": (135,206,235), "cyan": (0,255,255), "green":(0,128,0),
    "lime":(0,255,0), "olive":(128,128,0), "brown":(165,42,42),
    "white":(255,255,255), "silver":(192,192,192), "gray":(128,128,128),
    "black":(0,0,0)
}

NAMED_COLORS_LAB = {name: rgb_to_lab_cv(rgb) for name,rgb in NAMED_COLORS.items()}

def identify_named_color(rgb):
    lab = rgb_to_lab_cv(rgb)
    distances=[]
    for name,labref in NAMED_COLORS_LAB.items():
        d=ciede2000(lab,labref)
        distances.append((name,d,NAMED_COLORS[name]))
    distances.sort(key=lambda x: x[1])
    return distances[:3]

# -------------------- Seleção média de região --------------------
REGION_SIZE = 9

def average_region_color(img, x, y, size=REGION_SIZE):
    h,w = img.shape[:2]
    half = size//2
    x1,x2=max(0,x-half),min(w-1,x+half)
    y1,y2=max(0,y-half),min(h-1,y+half)
    region = img[y1:y2+1, x1:x2+1]
    if region.size == 0:
        return (0,0,0)
    return tuple(map(int, region.mean(axis=(0,1))))

# -------------------- Atualização do painel lateral --------------------
def update_info_panel(ref_color=None, comp_color=None, similarity=None, delta_e=None, named_candidates=None, reset=False):
    preview_ref.delete("all")
    preview_comp.delete("all")
    sim_canvas.delete("all")

    info_text.config(state="normal")
    info_text.delete("1.0",tk.END)

    if reset:
        info_text.insert(tk.END,"Nenhuma cor selecionada.\n")
        info_text.config(state="disabled")
        return

    if ref_color:
        preview_ref.create_rectangle(0,0,120,120, fill=rgb_to_hex(ref_color))
        preview_ref.create_text(60,130,text="Referência",fill="white")
    if comp_color:
        preview_comp.create_rectangle(0,0,120,120, fill=rgb_to_hex(comp_color))
        preview_comp.create_text(60,130,text="Comparada",fill="white")

    if similarity is not None:
        info_text.insert(tk.END,f"Similaridade: {similarity:.2f}%\n")
        info_text.insert(tk.END,f"ΔE00: {delta_e:.2f}\n\n")

        bar_width = int((similarity/100)*200)
        color = "#00C853" if similarity>80 else "#FFB300" if similarity>50 else "#D50000"
        sim_canvas.create_rectangle(0,0,bar_width,20,fill=color,outline="")
        sim_canvas.create_text(100,35,text=f"{similarity:.2f}%",fill="white")

    if named_candidates:
        info_text.insert(tk.END, "Top 3 cores próximas:\n")
        for name,d,rgb in named_candidates:
            info_text.insert(tk.END,f" • {name} — ΔE={d:.2f} — {rgb_to_hex(rgb)}\n")

    info_text.config(state="disabled")

# -------------------- Eventos --------------------
def on_click(event):
    global first_color_selected, ref_color_rgb, comp_color_rgb

    if app_image is None:
        return
    x,y=int(event.x),int(event.y)
    h,w=app_image.shape[:2]
    if x<0 or y<0 or x>=w or y>=h:
        return

    avg_rgb = average_region_color(app_image,x,y)

    if not first_color_selected:
        first_color_selected=True
        ref_color_rgb = avg_rgb
        update_info_panel(ref_color=ref_color_rgb)
        status_label.config(text="Clique agora na cor para comparar.")
    else:
        comp_color_rgb = avg_rgb
        lab1 = rgb_to_lab_cv(ref_color_rgb)
        lab2 = rgb_to_lab_cv(comp_color_rgb)
        de = ciede2000(lab1, lab2)
        sim = deltaE_to_similarity(de)
        named = identify_named_color(comp_color_rgb)

        update_info_panel(ref_color=ref_color_rgb, comp_color=comp_color_rgb,
                          similarity=sim, delta_e=de, named_candidates=named)

        status_label.config(text=f"Similaridade: {sim:.2f}%   ΔE={de:.2f}")
        compare_btn.config(state="normal")

def load_image():
    global app_image, displayed_photo, first_color_selected, ref_color_rgb, comp_color_rgb
    first_color_selected=False
    compare_btn.config(state="disabled")

    fp = filedialog.askopenfilename(filetypes=[("Imagens","*.jpg;*.jpeg;*.png;*.bmp")])
    if not fp:
        return

    img=cv2.imread(fp)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    h,w=img.shape[:2]
    scale=min(600/w,600/h)
    nw,nh=int(w*scale),int(h*scale)
    resized=cv2.resize(img,(nw,nh))

    canvas_bg=np.full((600,600,3),30,dtype=np.uint8)
    offx=(600-nw)//2
    offy=(600-nh)//2
    canvas_bg[offy:offy+nh, offx:offx+nw]=resized

    app_image=canvas_bg.copy()

    pil = Image.fromarray(app_image)
    displayed_photo = ImageTk.PhotoImage(pil)

    image_canvas.delete("all")
    image_canvas.create_image(0,0,anchor="nw",image=displayed_photo)

    status_label.config(text="Imagem carregada. Clique para escolher a referência.")
    update_info_panel(reset=True)

def reset_comparison():
    global first_color_selected, ref_color_rgb, comp_color_rgb
    first_color_selected=False
    compare_btn.config(state="disabled")
    update_info_panel(reset=True)
    status_label.config(text="Selecione uma nova referência.")

# -------------------- INTERFACE --------------------
app=tk.Tk()
app.title("GPSOFT — Comparador de Cores Avançado")
app.geometry("1000x720")
app.configure(bg="#2b2b2b")
app.resizable(False,False)

left_frame=tk.Frame(app,bg="#2b2b2b")
left_frame.pack(side="left",padx=12,pady=12)

image_canvas=tk.Canvas(left_frame,width=600,height=600,bg="#1e1e1e")
image_canvas.pack()
image_canvas.bind("<Button-1>",on_click)

load_btn=tk.Button(left_frame,text="📂 Carregar imagem",command=load_image,bg="#4a90e2",fg="white")
load_btn.pack(pady=6)

compare_btn=tk.Button(left_frame,text="🔄 Nova comparação",state="disabled",command=reset_comparison,bg="#4a90e2",fg="white")
compare_btn.pack()

status_label=tk.Label(left_frame,text="Aguardando imagem...",bg="#2b2b2b",fg="white")
status_label.pack(pady=6)

# Painel lateral (resultado ao lado da imagem)
right_frame=tk.Frame(app,bg="#2b2b2b")
right_frame.pack(side="left",padx=12,pady=12)

tk.Label(right_frame,text="Resultados",bg="#2b2b2b",fg="white",font=("Arial",14,"bold")).pack()

preview_container=tk.Frame(right_frame,bg="#2b2b2b")
preview_container.pack(pady=8)

preview_ref=tk.Canvas(preview_container,width=120,height=140,bg="#111")
preview_ref.grid(row=0,column=0,padx=6)

preview_comp=tk.Canvas(preview_container,width=120,height=140,bg="#111")
preview_comp.grid(row=0,column=1,padx=6)

sim_canvas=tk.Canvas(right_frame,width=200,height=60,bg="#111")
sim_canvas.pack(pady=8)

info_text=tk.Text(right_frame,width=40,height=15,bg="#111",fg="white")
info_text.pack()

# Variáveis globais
app_image=None
displayed_photo=None
first_color_selected=False
ref_color_rgb=None
comp_color_rgb=None

# Inicializar painel
update_info_panel(reset=True)

app.mainloop()
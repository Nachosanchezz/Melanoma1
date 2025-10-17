import os
import json
import math
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr
import cv2

# ======================
# Configuración y modelo
# ======================

# Ruta del checkpoint (puedes cambiarla desde el panel de "Files" del Space)
MODEL_PATH = os.environ.get("MODEL_PATH", "basic_cnn_minFN30.pt")  # o "basic_cnn_best.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debe coincidir con tu val/test transform
VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class BasicCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# Carga del checkpoint
def load_model_and_meta(model_path: str):
    model = BasicCNN(in_ch=3, num_classes=2).to(DEVICE)
    malignant_index = 1
    class_to_idx = { "Benign": 0, "Malignant": 1 }
    idx_to_class = {0: "Benign", 1: "Malignant"}

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE)
        # Distintos formatos posibles
        state = ckpt.get("model_state", ckpt if isinstance(ckpt, dict) else None)
        if state is None and isinstance(ckpt, dict):
            # A veces el checkpoint es el state_dict directamente
            state = ckpt
        model.load_state_dict(state)

        if isinstance(ckpt, dict):
            if "class_to_idx" in ckpt:
                class_to_idx = ckpt["class_to_idx"]
                idx_to_class = {v: k for k, v in class_to_idx.items()}
            malignant_index = ckpt.get("malignant_index", class_to_idx.get("Malignant", 1))
    else:
        print(f"[WARN] No se encontró {model_path}. Usa el panel de Files para subir tu .pt")

    model.eval()
    return model, class_to_idx, idx_to_class, malignant_index

MODEL, CLASS_TO_IDX, IDX_TO_CLASS, MAL_IDX = load_model_and_meta(MODEL_PATH)


# ==========================
# Utilidades de preprocesado
# ==========================

def to_tensor(img_pil: Image.Image) -> torch.Tensor:
    return VAL_TEST_TRANSFORMS(img_pil).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def infer(img_pil: Image.Image):
    x = to_tensor(img_pil)
    logits = MODEL(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs  # array shape [2], orden según IDX_TO_CLASS (0->Benign, 1->Malignant)


# ==========================
# Estimación de tamaño (px)
# ==========================

def estimate_lesion_size(img_pil: Image.Image):
    """
    Segmentación heurística para estimar el área y diámetro equivalente.
    Devuelve:
      - area_px
      - eq_diameter_px
      - overlay (PIL) con contorno
    """
    img_rgb = np.array(img_pil.convert("RGB"))  # HxWx3
    h, w = img_rgb.shape[:2]

    # Trabajamos sobre la imagen original (no reescalada) para medir en píxeles reales
    # 1) Suavizado
    blur = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=2, sigmaY=2)

    # 2) Convertimos a LAB y usamos cromas para detectar lesión
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    # Realzamos bordes/contraste ligero
    L_eq = cv2.equalizeHist(L)

    # 3) Mapa de "oscuridad" aproximado: invertimos L (lesiones suelen ser más oscuras)
    dark = cv2.normalize(255 - L_eq, None, 0, 255, cv2.NORM_MINMAX)

    # 4) Umbral de Otsu
    _, mask = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5) Morfología para limpiar ruido
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6) Nos quedamos con el componente conectado mayor
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        # No se encontró componente significativo
        area_px = 0
        eq_diam_px = 0.0
        overlay = Image.fromarray(img_rgb)
        return area_px, eq_diam_px, overlay

    # Ignora el fondo (label 0), busca comp. con mayor área
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    lesion_mask = (labels == max_idx).astype(np.uint8) * 255
    area_px = int(areas.max())

    # Diámetro equivalente (círculo con misma área)
    eq_diam_px = 2.0 * math.sqrt(area_px / math.pi)

    # Contorno para overlay
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_rgb.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)

    return area_px, float(eq_diam_px), Image.fromarray(overlay)


# ==============
# Lógica de UI
# ==============

def predict(image: Image.Image, threshold: float, mm_per_pixel: float):
    """
    - image: PIL
    - threshold: umbral para clasificar 'Malignant' según p_malignant >= threshold
    - mm_per_pixel: si > 0, convertimos tamaño a mm
    """
    if image is None:
        return {
            "error": "Sube una imagen (JPG/PNG)."
        }, None

    probs = infer(image)
    # Alinear índices con IDX_TO_CLASS
    # IDX_TO_CLASS: {0: "Benign", 1: "Malignant"} (u otro orden si venía del ckpt)
    p = { IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs)) }
    p_malignant = float(probs[MAL_IDX])
    p_benign = float(probs[1 - MAL_IDX])

    # Decisión por umbral (sobre prob. de "Malignant")
    is_malignant = p_malignant >= threshold
    pred_label = IDX_TO_CLASS[MAL_IDX] if is_malignant else IDX_TO_CLASS[1 - MAL_IDX]
    conf = p_malignant if is_malignant else p_benign

    # Tamaño
    area_px, eq_diam_px, overlay = estimate_lesion_size(image)

    result = {
        "prediccion": pred_label,
        "confianza": round(conf, 4),
        "umbral_usado": round(float(threshold), 3),
        "probabilidades": p,
        "tamano_pixeles": {
            "area_px": int(area_px),
            "diametro_equivalente_px": round(eq_diam_px, 2)
        }
    }

    # Conversión opcional a mm
    if mm_per_pixel and mm_per_pixel > 0:
        eq_diam_mm = eq_diam_px * mm_per_pixel
        # área en mm^2 ~ (px^2) * (mm/px)^2
        area_mm2 = area_px * (mm_per_pixel ** 2)
        result["tamano_mm"] = {
            "area_mm2": round(float(area_mm2), 2),
            "diametro_equivalente_mm": round(float(eq_diam_mm), 2),
            "mm_por_pixel": float(mm_per_pixel)
        }

    return result, overlay


DESCRIPTION = """
# Clasificador Melanoma (Benign vs Malignant)
Sube una imagen de una lesión cutánea y obtén la predicción del modelo (Benign/Malignant), la **confianza**, el **umbral** usado y una **estimación del tamaño**.

**Aviso:** Esta herramienta es educativa y no sustituye diagnóstico médico.
"""

with gr.Blocks(title="Melanoma Classifier") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Imagen del melanoma", sources=["upload", "clipboard"], height=320)
            thr_in = gr.Slider(0.3, 0.7, value=0.5, step=0.01, label="Umbral para 'Malignant' (p≥umbral)")
            mm_in  = gr.Number(value=0.0, label="mm por píxel (opcional, 0 si desconocido)")
            btn = gr.Button("Predecir", variant="primary")
        with gr.Column(scale=1):
            json_out = gr.JSON(label="Resultados")
            overlay_out = gr.Image(type="pil", label="Segmentación (estimada)", height=320)

    btn.click(fn=predict, inputs=[img_in, thr_in, mm_in], outputs=[json_out, overlay_out])

if __name__ == "__main__":
    demo.launch()

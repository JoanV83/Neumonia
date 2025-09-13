"""Coordinación de módulos para inferencia por CLI.

Flujo
-----
1) Lectura (DICOM/JPG/PNG) → RGB
2) Preprocesamiento → (1, 512, 512, 1)
3) Carga de modelo (.h5)
4) Grad-CAM → heatmap superpuesto (RGB)
5) Guardar resultados (por defecto en reports/figures) con cédula + timestamp
"""

from __future__ import annotations

# Silenciar logs de TF antes de importarlo indirectamente
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.data.preprocess_img import preprocess_image
from src.data.read_img import read_dicom, read_image
from src.models.grad_cam import grad_cam
from src.models.load_model import load_model

LABELS = ("bacteriana", "normal", "viral")


def _safe_id(s: Optional[str]) -> str:
    """Sanitiza un ID/cédula para usarlo en nombres de archivo."""
    if not s:
        return ""
    s = s.strip()
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))


def run_pipeline(
    input_path: str,
    model_path: str,
    last_conv: str,
) -> Tuple[str, float, np.ndarray]:
    """Ejecuta el pipeline completo y devuelve (label, prob, heatmap_rgb).

    Parameters
    ----------
    input_path : str
        Ruta a la imagen de entrada (DICOM/JPG/PNG).
    model_path : str
        Ruta al modelo (.h5).
    last_conv : str
        Nombre de la última capa convolucional.

    Returns
    -------
    tuple
        (label_str, prob_float, heatmap_rgb_uint8)
    """
    # Lectura (detecta por extensión)
    if input_path.lower().endswith(".dcm"):
        rgb, _ = read_dicom(input_path)
    else:
        rgb, _ = read_image(input_path)

    # Preprocesamiento
    x = preprocess_image(rgb)

    # Modelo y Grad-CAM
    model = load_model(model_path)
    class_idx, prob, heatmap_rgb = grad_cam(
        model=model, x=x, last_conv_name=last_conv, base_rgb=rgb
    )

    # Etiqueta legible
    label = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)
    return label, prob, heatmap_rgb


def save_outputs(
    outdir: str,
    label: str,
    prob: float,
    heatmap_rgb: np.ndarray,
    patient_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Guarda resultados (texto + heatmap) y retorna sus rutas.

    Crea nombres únicos con cédula (si se proporciona) + timestamp.

    Parameters
    ----------
    outdir : str
        Directorio de salida.
    label : str
        Clase predicha (texto).
    prob : float
        Probabilidad (0–1).
    heatmap_rgb : np.ndarray
        Imagen RGB (512, 512, 3) uint8.
    patient_id : str | None
        Cédula/ID del paciente para prefijar nombres.

    Returns
    -------
    tuple[str, str]
        (ruta_heatmap_png, ruta_resultado_txt)
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    pid = _safe_id(patient_id)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{pid}_" if pid else ""

    heatmap_path = out / f"{prefix}heatmap_{stamp}.png"
    txt_path = out / f"{prefix}resultado_{stamp}.txt"

    # OpenCV espera BGR al escribir
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"clase={label}, probabilidad={prob:.4f}\n")

    return str(heatmap_path), str(txt_path)


# ---- Alias para compatibilidad con smoke ---------------------- #
def predict_with_explain(
    dicom_path: str,
    model_path: str,
    last_conv_name: str,
) -> Tuple[str, float, np.ndarray]:
    """Alias de compatibilidad: llama a `run_pipeline`."""
    return run_pipeline(
        input_path=dicom_path, model_path=model_path, last_conv=last_conv_name
    )
# ------------------------------------------------------------------------- #


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de detección de neumonía con Grad-CAM."
    )
    parser.add_argument("--input", required=True, help="Ruta a imagen (DICOM/JPG/PNG)")
    parser.add_argument("--model", required=True, help="Ruta a modelo .h5")
    parser.add_argument(
        "--last-conv",
        default="conv10_thisone",
        help="Nombre de la última capa convolucional del modelo",
    )
    parser.add_argument(
        "--outdir",
        default="reports/figures",
        help="Directorio de salida (por defecto reports/figures)",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Cédula/ID del paciente para prefijar los archivos de salida",
    )
    args = parser.parse_args()

    label, prob, heatmap = run_pipeline(
        input_path=args.input, model_path=args.model, last_conv=args.last_conv
    )
    heatmap_path, txt_path = save_outputs(
        outdir=args.outdir,
        label=label,
        prob=prob,
        heatmap_rgb=heatmap,
        patient_id=args.patient_id,
    )

    print(f"Clase: {label}  Probabilidad: {prob:.2%}")
    print("Archivos guardados en:")
    print(f"  {heatmap_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
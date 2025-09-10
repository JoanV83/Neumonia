"""Coordinación de módulos para inferencia por línea de comandos.

Responsabilidades:
- Unificar la ejecución: lectura → preprocesamiento → modelo → Grad-CAM.
- Retornar/guardar clase, probabilidad y heatmap para la interfaz/CLI.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import cv2

from src.read_img import read_dicom, read_image
from src.preprocess_img import preprocess_image
from src.load_model import load_model
from src.grad_cam import grad_cam


LABELS = ("bacteriana", "normal", "viral")  # ajusta si tu modelo difiere


def run_pipeline(input_path: str,
                 model_path: str,
                 last_conv: str) -> Tuple[str, float, np.ndarray]:
    """Ejecuta el pipeline completo y devuelve resultados.

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


def save_outputs(outdir: str,
                 label: str,
                 prob: float,
                 heatmap_rgb: np.ndarray) -> None:
    """Guarda resultados (texto + heatmap) en disco.

    Parameters
    ----------
    outdir : str
        Directorio de salida.
    label : str
        Clase predicha (texto).
    prob : float
        Probabilidad (0–1).
    heatmap_rgb : np.ndarray
        Imagen de heatmap RGB (512, 512, 3) uint8.
    """
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, "heatmap.png"),
                cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    with open(os.path.join(outdir, "resultado.txt"), "w", encoding="utf-8") as f:
        f.write(f"clase={label}, probabilidad={prob:.4f}\n")


def main() -> None:
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
    parser.add_argument("--outdir", default="reports", help="Directorio de salida")
    args = parser.parse_args()

    label, prob, heatmap = run_pipeline(
        input_path=args.input, model_path=args.model, last_conv=args.last_conv
    )
    save_outputs(args.outdir, label, prob, heatmap)

    print(f"Clase: {label}  Probabilidad: {prob:.2%}")
    print(f"Archivos guardados en: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()

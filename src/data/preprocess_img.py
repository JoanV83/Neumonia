"""Preprocesamiento de imágenes para el modelo.

Pasos
-----
1) Escala de grises
2) Redimensionado a 512x512
3) CLAHE
4) Normalización [0, 1]
5) Añadir dims (batch y canal) → (1, 512, 512, 1)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess_image(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Preprocesa una imagen a tensor listo para el modelo.

    Parameters
    ----------
    img_rgb : np.ndarray
        Imagen RGB (H, W, 3) o monocanal (H, W).
    target_size : tuple[int, int], default=(512, 512)
        Tamaño deseado (ancho, alto).

    Returns
    -------
    np.ndarray
        Tensor con forma (1, 512, 512, 1), dtype=float32 y rango [0, 1].
    """
    # A escala de grises
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    # Redimensionado
    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # CLAHE requiere uint8
    if gray.dtype != np.uint8:
        vmin = float(gray.min())
        vmax = float(gray.max()) + 1e-8
        gray = ((gray - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Normalización y shape final
    x = gray.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))  # (1, H, W, 1)
    return x

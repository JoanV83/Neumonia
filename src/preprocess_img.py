"""Preprocesamiento de imágenes para el modelo.

Responsabilidades:
- Redimensionar a 512×512.
- Convertir a escala de grises.
- Aplicar CLAHE.
- Normalizar a [0, 1].
- Convertir a tensor con forma (1, 512, 512, 1).
"""

from typing import Tuple
import numpy as np
import cv2


def preprocess_image(img_rgb: np.ndarray,
                     target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Preprocesa una imagen RGB a tensor listo para el modelo.

    Parameters
    ----------
    img_rgb : np.ndarray
        Imagen en RGB (H, W, 3) o monocanal (H, W).
    target_size : tuple[int, int], optional
        Tamaño destino, por defecto (512, 512).

    Returns
    -------
    np.ndarray
        Tensor con forma (1, 512, 512, 1) y dtype=float32 en [0, 1].
    """
    # A escala de grises
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    # Resize
    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # CLAHE (mejora contraste local)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Si viene flotante, llevar a 0-255 uint8 antes de CLAHE
    if gray.dtype != np.uint8:
        vmin, vmax = float(gray.min()), float(gray.max()) + 1e-8
        gray = ((gray - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    gray = clahe.apply(gray)

    # Normalizar y dar forma (batch, H, W, C)
    x = gray.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))  # (1, 512, 512, 1)
    return x

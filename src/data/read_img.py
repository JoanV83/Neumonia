"""Utilidades de lectura de imágenes médicas y de disco.

Responsabilidades
-----------------
- Leer imágenes DICOM y convertirlas a NumPy RGB + PIL.
- Leer imágenes comunes (PNG/JPG) y devolver RGB + PIL.

Notas
-----
- En memoria trabajamos en **RGB**.
- Si el DICOM trae WindowCenter/WindowWidth, se aplica windowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


def _to_uint8_0_255(arr: np.ndarray) -> np.ndarray:
    """Normaliza un arreglo a uint8 [0, 255] de forma robusta."""
    arr = arr.astype(np.float32, copy=False)
    vmax = float(arr.max()) if arr.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (np.maximum(arr, 0.0) / (vmax + 1e-8)) * 255.0
    return out.astype(np.uint8)


def _apply_windowing(ds: "dicom.dataset.FileDataset",
                     img: np.ndarray) -> np.ndarray:
    """Aplica WindowCenter/WindowWidth si están disponibles en el DICOM."""
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return _to_uint8_0_255(img)

    if isinstance(wc, (list, tuple)):
        wc = wc[0]
    if isinstance(ww, (list, tuple)):
        ww = ww[0]
    wc = float(wc)
    ww = float(ww) if float(ww) != 0 else 1.0

    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img.astype(np.float32), lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return (img * 255.0).astype(np.uint8)


def read_dicom(path: Union[str, Path]) -> Tuple[np.ndarray, Image.Image]:
    """Lee una imagen DICOM y la devuelve como RGB + objeto PIL.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo DICOM (.dcm).

    Returns
    -------
    tuple[np.ndarray, PIL.Image.Image]
        (img_rgb, pil_img) donde img_rgb es (H, W, 3) uint8.
    """
    p = Path(path)
    ds = dicom.dcmread(str(p))
    arr = ds.pixel_array

    img_u8 = _apply_windowing(ds, arr)

    # Manejo de dimensionalidad/canales:
    if img_u8.ndim == 2:
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.ndim == 3:
        if img_u8.shape[2] == 3:
            # Conserva tal cual (ya sería RGB).
            img_rgb = img_u8.copy()
        else:
            # Toma el primer canal y lo convierte a RGB
            img_rgb = cv2.cvtColor(img_u8[..., 0], cv2.COLOR_GRAY2RGB)
    else:
        # Volúmenes: toma la primera “slice”
        base = img_u8
        while base.ndim > 2:
            base = base[0]
        img_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(img_rgb)
    return img_rgb, pil_img


def read_image(path: Union[str, Path]) -> Tuple[np.ndarray, Image.Image]:
    """Lee una imagen común (JPG/PNG) y devuelve RGB + PIL.

    Parameters
    ----------
    path : str | Path
        Ruta a imagen común.

    Returns
    -------
    tuple[np.ndarray, PIL.Image.Image]
        (img_rgb, pil_img) con img_rgb (H, W, 3) uint8.
    """
    p = Path(path)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, Image.fromarray(rgb)

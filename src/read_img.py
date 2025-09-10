"""Utilidades de lectura de imágenes médicas y de disco.

Responsabilidades:
- Leer imágenes DICOM y convertir a arreglos NumPy.
- Opcionalmente devolver una versión PIL para visualizar en interfaz.
"""

from typing import Tuple
import numpy as np
import cv2
import pydicom as dicom
from PIL import Image


def read_dicom(path: str) -> Tuple[np.ndarray, Image.Image]:
    """Lee una imagen DICOM y la devuelve en RGB y como PIL.

    Convierte el pixel data a rango [0, 255] uint8 y a RGB para
    facilitar su visualización en interfaces gráficas.

    Parameters
    ----------
    path : str
        Ruta al archivo DICOM (.dcm).

    Returns
    -------
    np.ndarray
        Imagen en formato RGB (H, W, 3) dtype=uint8.
    PIL.Image.Image
        Objeto PIL para mostrar en GUI.
    """
    ds = dicom.dcmread(path)
    img = ds.pixel_array.astype(float)

    # Normalización robusta a [0, 255]
    img = (np.maximum(img, 0) / (img.max() + 1e-8)) * 255.0
    img_u8 = img.astype(np.uint8)

    # A RGB para visualización (muchos DICOM son 1 canal)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    return img_rgb, pil_img


def read_image(path: str) -> Tuple[np.ndarray, Image.Image]:
    """Lee JPG/PNG y devuelve RGB + PIL.

    Parameters
    ----------
    path : str
        Ruta a imagen común (jpg, png, etc.).

    Returns
    -------
    np.ndarray
        Imagen en formato RGB (H, W, 3) dtype=uint8.
    PIL.Image.Image
        Objeto PIL para mostrar en GUI.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, Image.fromarray(rgb)
